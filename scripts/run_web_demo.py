import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_qa import BM25Retriever, HybridRetriever, LegalQAPipeline, load_documents
from legal_qa.categories import CATEGORY_LABELS
from legal_qa.reranking import CrossEncoderReranker


FINAL_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


HTML = """<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<title>ระบบถามตอบกฎหมายไทย</title>
<style>
body { font-family: sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
textarea { width: 100%; min-height: 90px; padding: .75rem; }
button { margin-top: .75rem; padding: .6rem 1rem; cursor: pointer; }
pre { white-space: pre-wrap; background: #f4f4f4; padding: 1rem; }
.source { border-left: 4px solid #555; padding: .5rem 1rem; margin: 1rem 0; }
</style>
</head>
<body>
<h1>ระบบถามตอบกฎหมายไทย</h1>
<p>ระบบค้นหาหลักฐานทางกฎหมาย จัดอันดับ และเตรียมคำถามพร้อมหลักฐาน</p>
<select id="category">{category_options}</select>
<textarea id="question" placeholder="เช่น ถ้าขโมยของคนอื่น มีความผิดอะไร"></textarea>
<br><button onclick="ask()">ค้นหาข้อมูล</button>
<h2>ผลลัพธ์</h2><div id="output"></div>
<script>
async function ask() {
  const question = document.getElementById('question').value;
    const response = await fetch('/api/ask', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question, category: document.getElementById('category').value})
  });
  const data = await response.json();
  const output = document.getElementById('output');
  if (data.error) { output.textContent = data.error; return; }
      output.innerHTML = '<h3>สถานะคำตอบ</h3><p>' + data.answer_status + '</p>' +
        '<h3>แหล่งข้อมูล</h3>' + data.sources.map(source =>
            '<div class="source"><b>' + source.citation + '</b>' +
            '<p>คะแนน: ' + source.score.toFixed(4) + '</p>' +
            '<p>' + source.context + '</p></div>').join('') +
          '<h3>คำถามพร้อมหลักฐาน</h3><pre>' + data.prompt + '</pre>';
}
</script>
</body></html>"""


def make_handler(pipeline):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/":
                self.send_error(404)
                return
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/api/ask":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            question = str(payload.get("question", "")).strip()
            if not question:
                self._json({"error": "กรุณาพิมพ์คำถาม"}, status=400)
                return
            category = str(payload.get("category", "all"))
            response = pipeline.answer(question, top_k=5, category=category)
            self._json({
                "answer": response.answer,
                "answer_status": (
                    "ยังไม่ได้เชื่อมต่อเครื่องมือสร้างคำตอบ จึงแสดงหลักฐานและคำถามพร้อมหลักฐานแทน"
                    if response.answer is None
                    else "สร้างคำตอบจากบริบทที่มีหลักฐานอ้างอิงแล้ว"
                ),
                "prompt": response.prompt,
                "sources": [
                    {
                        "citation": (
                            f"{result.document.law_title} "
                            f"มาตรา {result.document.section}"
                        ),
                        "law_title": result.document.law_title,
                        "section": result.document.section,
                        "context": result.document.context,
                        "score": result.score,
                    }
                    for result in response.sources
                ],
            })

        def _json(self, payload, status=200):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                # The client may close a request while the model is responding.
                pass

        def log_message(self, format, *args):
            return

    return Handler


def parse_args():
    parser = argparse.ArgumentParser(description="เรียกใช้เดโมระบบถามตอบกฎหมายไทยบนเว็บ")
    parser.add_argument("--documents", default="data/processed/legal_documents.csv")
    parser.add_argument("--retriever", choices=("bm25", "hybrid"), default="hybrid")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--embeddings", default="results/minilm_document_embeddings.pt")
    parser.add_argument("--reranker-model", default=FINAL_RERANKER_MODEL)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main():
    args = parse_args()
    documents = load_documents(args.documents)
    if args.retriever == "hybrid":
        retriever = HybridRetriever(
            documents,
            alpha=args.alpha,
            embeddings_path=args.embeddings,
        )
    else:
        retriever = BM25Retriever(documents)

    reranker = None
    if not args.no_reranker:
        reranker = CrossEncoderReranker(args.reranker_model)
    pipeline = LegalQAPipeline(retriever, reranker=reranker)
    category_options = "".join(
        f'<option value="{key}">{label}</option>'
        for key, label in CATEGORY_LABELS.items()
    )
    global HTML
    HTML = HTML.replace("{category_options}", category_options)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(pipeline))
    print(f"เปิดเดโมบนเว็บที่ http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()