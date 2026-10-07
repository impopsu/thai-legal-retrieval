import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_qa import BM25Retriever, HybridRetriever, SemanticRetriever, load_documents
from legal_qa.reranking import CrossEncoderReranker


FINAL_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
METHODS = ("bm25", "minilm", "hybrid", "bge_m3", "hybrid_reranker")


class PeakRssMonitor:
    def __init__(self, interval=0.01):
        self.process = psutil.Process()
        self.interval = interval
        self.peak_bytes = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _record(self):
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)

    def _sample(self):
        while not self.stop_event.is_set():
            self._record()
            self.stop_event.wait(self.interval)
        self._record()

    def start(self):
        self._record()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()


def _create_retriever(method, documents, args):
    if method == "bm25":
        return BM25Retriever(documents), None
    if method == "minilm":
        return SemanticRetriever(
            documents,
            embeddings_path=args.minilm_embeddings,
        ), None
    if method == "hybrid":
        return HybridRetriever(
            documents,
            embeddings_path=args.minilm_embeddings,
            alpha=args.alpha,
        ), None
    if method == "bge_m3":
        return SemanticRetriever(
            documents,
            model_name="BAAI/bge-m3",
            embeddings_path=args.bge_embeddings,
            batch_size=args.bge_batch_size,
        ), None
    retriever = HybridRetriever(
        documents,
        embeddings_path=args.minilm_embeddings,
        alpha=args.alpha,
    )
    reranker = CrossEncoderReranker(
        FINAL_RERANKER_MODEL,
        batch_size=args.reranker_batch_size,
    )
    return retriever, reranker


def _retrieve(retriever, reranker, question):
    if reranker is None:
        return retriever.search(question, top_k=5)
    candidates = retriever.search(question, top_k=20)
    return reranker.rerank(question, candidates, top_k=5)


def _run_worker(method, args):
    monitor = PeakRssMonitor()
    monitor.start()
    documents = load_documents(args.documents)
    questions = pd.read_parquet(args.questions)["question"].head(args.limit).tolist()
    if not questions:
        raise ValueError("No questions available for runtime benchmark")
    retriever, reranker = _create_retriever(method, documents, args)

    _retrieve(retriever, reranker, questions[0])
    latencies = []
    for question in questions:
        started = time.perf_counter()
        _retrieve(retriever, reranker, question)
        latencies.append((time.perf_counter() - started) * 1000)
    monitor.stop()

    return {
        "method": method,
        "avg_latency_ms": sum(latencies) / len(latencies),
        "peak_memory_mb": monitor.peak_bytes / (1024 ** 2),
        "num_questions": len(questions),
    }


def _worker_command(method, args):
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        method,
        "--documents",
        args.documents,
        "--questions",
        args.questions,
        "--limit",
        str(args.limit),
        "--minilm-embeddings",
        args.minilm_embeddings,
        "--bge-embeddings",
        args.bge_embeddings,
        "--alpha",
        str(args.alpha),
        "--bge-batch-size",
        str(args.bge_batch_size),
        "--reranker-batch-size",
        str(args.reranker_batch_size),
    ]


def main():
    parser = argparse.ArgumentParser(description="Benchmark retriever latency and peak RSS")
    parser.add_argument("--worker", choices=METHODS)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )
    parser.add_argument("--documents", default="data/processed/legal_documents_categorized.csv")
    parser.add_argument("--questions", default="data/processed/validation_retrieval.parquet")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--minilm-embeddings", default="results/minilm_document_embeddings.pt")
    parser.add_argument("--bge-embeddings", default="/tmp/thai_legal_bge_m3_embeddings.pt")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--bge-batch-size", type=int, default=4)
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument("--output", default="results/runtime_benchmark.csv")
    args = parser.parse_args()

    if args.worker:
        print(json.dumps(_run_worker(args.worker, args)))
        return

    rows = []
    for method in args.methods:
        print(f"Benchmarking {method} on {args.limit} questions", flush=True)
        result = subprocess.run(
            _worker_command(method, args),
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(
                f"{method} benchmark failed:\n{result.stderr or result.stdout}"
            )
        output_lines = [line for line in result.stdout.splitlines() if line.strip()]
        rows.append(json.loads(output_lines[-1]))
        print(rows[-1])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows,
        columns=["method", "avg_latency_ms", "peak_memory_mb", "num_questions"],
    ).to_csv(output_path, index=False)
    print(f"Saved runtime benchmark to {output_path}")


if __name__ == "__main__":
    main()