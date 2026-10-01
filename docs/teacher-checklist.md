# องค์ประกอบสำหรับนำเสนออาจารย์

## 1. Dataset ที่จัดเตรียมใหม่

```text
data/processed/qa_records.parquet
data/processed/legal_documents_categorized.csv
```

`qa_records.parquet` เก็บ question, positive context, positive answer,
metadata และ category แบบหนึ่งแถวต่อหนึ่ง positive context

สร้างใหม่ด้วย:

```bash
python scripts/prepare_qa_dataset.py
```

## 2. การเลือกหมวดกฎหมาย

CLI:

```bash
python scripts/run_qa_demo.py --category property
```

Web demo:

```bash
python scripts/run_web_demo.py \
  --documents data/processed/legal_documents_categorized.csv
```

หมวดปัจจุบันมีทรัพย์สิน, อาญา, ฆาตกรรม/ชีวิตและร่างกาย, การเงิน, บริษัท,
แรงงาน, ครอบครัว/มรดก, ภาษี, วิธีพิจารณา และ other

## 3. Context evaluation

```bash
python scripts/evaluate_context.py \
  --questions data/processed/validation_retrieval.parquet \
  --documents data/processed/legal_documents_categorized.csv \
  --limit 100
```

Metrics คือ ContextRecall@1/3/5, ContextPrecision@5 และ ContextMRR@5

## 4. Answer evaluation

เมื่อมีคำตอบจาก LLM เป็น JSONL:

```bash
python scripts/evaluate_answers.py \
  --predictions results/qa_predictions.jsonl \
  --references data/raw/test-00000-of-00001.parquet
```

Metrics คือ answer token F1, citation correctness และ abstention

สร้างคำตอบ validation 100 ข้อด้วย Gemini 2.5 Flash-Lite:

```bash
python scripts/generate_gemini_answers.py \
  --input results/qa_prompts_validation_100.jsonl \
  --output results/qa_predictions_validation_100.jsonl \
  --model gemini-2.5-flash-lite
```

## 5. Hardware

```bash
python scripts/check_hardware.py
```

เครื่องปัจจุบันไม่มี NVIDIA CUDA GPU จึงใช้ CPU fallback

## ข้อจำกัดที่ควรอธิบาย

- category เป็น rule-based label และควรตรวจทานหากใช้เป็น gold label
- answer F1 ต้องรันหลังมีคำตอบจาก LLM จริง
- aggregate retrieval score อาจสูง แต่คำถามภาษาพูดบางข้อยังดึง evidence อ่อน
- official test ห้ามใช้เลือก parameter หรือ category rules