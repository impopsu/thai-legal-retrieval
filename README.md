# Thai Legal QA

การทำ legal retrieval และ grounded question answering ภาษาไทยโดยใช้ BM25,
multilingual semantic search, hybrid retrieval และ reranking

## สถานะปัจจุบัน

ส่วน core ที่นำกลับมาใช้ซ้ำได้อยู่ใน `legal_qa/` ส่วนการทดลองเดิมยังอยู่ใน
`experiments/` เพื่อใช้เปรียบเทียบ final pipeline ที่เลือกคือ Hybrid MiniLM ที่มี
alpha `0.5`, candidate top-20 และ Cross-Encoder reranking เหลือ evidence top-5
ส่วน BGE-M3 และ Hybrid BGE-M3 เป็นการทดลองเปรียบเทียบใน validation

hybrid score convention คือ:

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

ดู [docs/project-summary.md](docs/project-summary.md) สำหรับ dataset, methods,
metrics, final results และข้อจำกัด

เตรียม question-level positive contexts และ categories:

```bash
python scripts/prepare_qa_dataset.py
```

ประเมิน retrieved contexts เทียบกับ `positive_contexts`:

```bash
python scripts/evaluate_context.py \
	--questions data/processed/validation_retrieval.parquet \
	--documents data/processed/legal_documents_categorized.csv \
	--limit 100
```

ตรวจ GPU และสร้าง grounded prompts/คำตอบ Gemini ได้ด้วย scripts ที่ระบุใน
`docs/project-summary.md` โดยต้องตั้ง `GEMINI_API_KEY` ใน terminal เท่านั้น

```bash
python scripts/export_qa_prompts.py \
	--questions data/processed/validation_retrieval.parquet \
	--limit 100 \
	--output results/qa_prompts_validation_100.jsonl
```

Generate answers with Gemini 3.5 Flash-Lite:

```bash
python scripts/generate_gemini_answers.py \
	--input results/qa_prompts_validation_100.jsonl \
	--output results/qa_predictions_validation_100.jsonl \
	--model gemini-3.5-flash-lite \
	--retry-errors
```

Evaluate answer F1 and citations:

```bash
python scripts/evaluate_answers.py \
	--predictions results/qa_predictions_validation_100.jsonl \
	--references data/processed/validation_retrieval.parquet \
	--output results/answer_metrics_validation_100.csv
```

The generator resumes successful records and logs failed records without
printing the API key.
