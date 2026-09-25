# Thai Legal QA

การทำ legal retrieval และ grounded question answering ภาษาไทยโดยใช้ BM25,
multilingual semantic search, hybrid retrieval และ reranking

## สถานะปัจจุบัน

ส่วน core ที่นำกลับมาใช้ซ้ำได้อยู่ใน `legal_qa/` ส่วนการทดลองเดิมยังอยู่ใน
`experiments/` เพื่อใช้เปรียบเทียบ production pipeline สุดท้ายที่เลือกคือ Hybrid
MiniLM ที่มี alpha `0.5` ตามด้วย Cross-Encoder reranking ส่วน BGE-M3 และ Hybrid
BGE-M3 ยังคงเป็นการทดลองเปรียบเทียบใน validation โดยกำหนด hybrid score ดังนี้:

```text
hybrid_score = alpha * normalized_bm25 + (1 - alpha) * normalized_semantic
```

การสร้างคำตอบเป็นทางเลือก และจะโหลด language model เฉพาะเมื่อมีการระบุอย่างชัดเจน

## รัน retrieval evaluation

```bash
python scripts/evaluate_retrieval.py --methods bm25
```

เปรียบเทียบ retrievers บน validation ด้วย protocol เดียวกัน:

```bash
python scripts/benchmark_retrievers.py \
	--split validation \
	--methods bm25 minilm hybrid_minilm
```

ประเมิน hybrid retriever และ cache MiniLM document embeddings:

```bash
python scripts/evaluate_retrieval.py --methods bm25 hybrid --alpha 0.5
```

ประเมิน cross-encoder reranker ที่เป็นทางเลือกบน validation data:

```bash
python scripts/evaluate_reranker.py \
	--model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

## รัน QA demo

```bash
python scripts/run_qa_demo.py
```

demo เริ่มต้นใช้ final pipeline ที่เลือกไว้: Hybrid MiniLM alpha `0.5`, candidate
top-20 และ Cross-Encoder reranking เป็น top-5 ใช้ `--no-reranker` สำหรับ baseline
ที่เร็วขึ้น หากไม่มี generator model demo จะแสดง legal evidence ที่จัดอันดับแล้ว
และ grounded prompt ใช้ `--generator-model` เพื่อระบุ Hugging Face text-to-text
model ที่เข้ากันได้สำหรับการสร้างคำตอบ

ดู [docs/roadmap.md](docs/roadmap.md) สำหรับ scope, metrics และ milestones ถัดไป
ดู [docs/dataset.md](docs/dataset.md) สำหรับแหล่งที่มาของ dataset, papers และ
evaluation protocol ที่ไม่แตะต้อง test
ดู [docs/p0-status.md](docs/p0-status.md) สำหรับ retrieval benchmark ปัจจุบัน
และ blockers ที่ยังเหลืออยู่
ดู [docs/human-evaluation.md](docs/human-evaluation.md) สำหรับ rubric การตรวจคำตอบ
และ [docs/reproducibility.md](docs/reproducibility.md) สำหรับขั้นตอนการตั้งค่า

เมื่อมีคำตอบที่สร้างแล้วในรูปแบบ JSONL ให้ประเมินด้วย:

```bash
python scripts/evaluate_answers.py --predictions results/qa_predictions.jsonl
```

ส่งออก grounded prompts สำหรับ external หรือ future LLM:

```bash
python scripts/export_qa_prompts.py --limit 100
```

สร้างคำตอบด้วย Gemini ให้ตั้งค่า `GEMINI_API_KEY` โดยตรงใน terminal
และอย่าใส่ค่าไว้ในไฟล์หรือ command-line argument:

```bash
python scripts/export_qa_prompts.py \
	--questions data/processed/validation_retrieval.parquet \
	--limit 100 \
	--output results/qa_prompts_validation_100.jsonl

python scripts/generate_gemini_answers.py \
	--input results/qa_prompts_validation_100.jsonl \
	--output results/qa_predictions_validation_100.jsonl \
	--model gemini-3.1-pro-preview

python scripts/evaluate_answers.py \
	--predictions results/qa_predictions_validation_100.jsonl \
	--references data/processed/validation_retrieval.parquet \
	--output results/answer_metrics_validation_100.csv
```
