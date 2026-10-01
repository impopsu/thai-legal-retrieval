# สรุปโครงงานระบบถามตอบและค้นหากฎหมายไทย / RAG

## 1. เป้าหมายโครงงาน

ระบบรับคำถามกฎหมายภาษาไทย ค้นหาบริบทกฎหมายที่เกี่ยวข้อง จัดอันดับด้วย
reranker และเตรียมหลักฐานสำหรับ LLM โดยคำตอบต้องอ้างอิงจากหลักฐานเท่านั้น

```text
คำถาม
-> Hybrid Retrieval
-> ผู้สมัคร 20 อันดับแรก
-> Cross-Encoder Reranker
-> หลักฐาน 5 อันดับแรก
-> พรอมต์ LLM ที่มีหลักฐานอ้างอิง
-> คำตอบพร้อมการอ้างอิง
```

## 2. ชุดข้อมูล

ชื่อ: **WangchanX-Legal-ThaiCCL-RAG**

แหล่งข้อมูล: https://huggingface.co/datasets/airesearch/WangchanX-Legal-ThaiCCL-RAG

บทความอ้างอิง:

- NitiBench: https://aclanthology.org/2025.emnlp-main.1739/
- A Free Format Legal Question Answering System: https://aclanthology.org/2021.nllp-1.11/

ข้อมูลภายในเครื่อง:

```text
data/raw/train-00000-of-00001.parquet
data/raw/test-00000-of-00001.parquet
```

การแบ่งข้อมูลปัจจุบัน:

```text
Train: 8,211 questions
Validation: 1,643 questions
Test: 3,742 questions
```

แต่ละระเบียนประกอบด้วย `question`, `positive_contexts`,
`hard_negative_contexts`, `positive_answer` และ `hard_negative_answer`

เตรียมระเบียนระดับคำถามและเอกสารที่จัดหมวดหมู่แล้ว:

```bash
python scripts/prepare_qa_dataset.py
```

ไฟล์ผลลัพธ์:

```text
data/processed/qa_records.parquet
data/processed/legal_documents_categorized.csv
```

ระเบียน QA ประกอบด้วยคำถาม บริบทที่ถูกต้อง คำตอบที่ถูกต้อง ข้อมูลกำกับกฎหมาย
และหมวดหมู่ โดยมีหนึ่งแถวต่อหนึ่งบริบทที่ถูกต้อง

## 3. หมวดหมู่กฎหมาย

ชุดข้อมูลไม่มีฟิลด์หมวดหมู่อย่างเป็นทางการ โครงงานใช้ตัวจำแนกแบบอิงกฎที่ขยายได้
จากคำถาม ชื่อกฎหมาย และบริบท ตัวจำแนกนี้ใช้เป็นตัวกรองการค้นหา ไม่ใช่ป้ายกำกับอ้างอิง

```text
property   กฎหมายทรัพย์สิน
criminal   กฎหมายอาญา
murder     กฎหมายฆาตกรรม/ชีวิตและร่างกาย
financial  กฎหมายการเงิน
company    กฎหมายบริษัทและนิติบุคคล
labor      กฎหมายแรงงาน
family     กฎหมายครอบครัวและมรดก
tax        กฎหมายภาษีอากร
procedure  กฎหมายวิธีพิจารณา
other      กฎหมายหมวดอื่น
```

สามารถเพิ่มกฎได้ใน `legal_qa/categories.py`

ตัวอย่างการใช้หมวดหมู่:

```bash
python scripts/run_qa_demo.py --category property
python scripts/run_web_demo.py --documents data/processed/legal_documents_categorized.csv
```

## 4. การตั้งค่าการค้นหาขั้นสุดท้าย

```text
Retriever: Hybrid MiniLM
Alpha: 0.5
Candidates: top-20
Reranker: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
Final evidence: top-5
```

ไม่ใช้ชุดทดสอบอย่างเป็นทางการในการเลือกค่า alpha, top-k, โมเดล หรือ reranker

## 5. ผลการค้นหา

ผลการทดสอบเปรียบเทียบทั้งหมดบนชุดตรวจสอบ:

| Method | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---|---:|---:|---:|---:|
| BM25 | 0.4948 | 0.6519 | 0.7158 | 0.5783 |
| Semantic MiniLM | 0.3214 | 0.4729 | 0.5508 | 0.4061 |
| Hybrid MiniLM | 0.5673 | 0.7206 | 0.7669 | 0.6462 |
| BGE-M3 | 0.6543 | 0.8113 | 0.8673 | 0.7379 |
| Hybrid BGE-M3 | 0.6409 | 0.7815 | 0.8411 | 0.7166 |

ผล reranker ทั้งหมดบนชุดตรวจสอบ:

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---|---:|---:|---:|---:|
| Hybrid MiniLM baseline | 0.5673 | 0.7206 | 0.7669 | 0.6462 |
| Hybrid MiniLM + reranker | 0.6999 | 0.8223 | 0.8497 | 0.7618 |

ผลการค้นหาบนชุดทดสอบที่ไม่ได้นำไปปรับแต่ง:

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---|---:|---:|---:|---:|
| Hybrid MiniLM baseline | 0.5882 | 0.7435 | 0.7990 | 0.6701 |
| Hybrid MiniLM + reranker | 0.7194 | 0.8354 | 0.8626 | 0.7790 |

แถว BM25 และ Semantic MiniLM ในตาราง validation ยืนยันได้จาก
`results/retrieval_benchmark_validation_bm25_minilm.csv`

## 6. การประเมินบริบท

เปรียบเทียบบริบทที่ค้นพบกับ `positive_contexts` โดยใช้ `unique_key` ของเอกสาร

คำสั่งรัน:

```bash
python scripts/evaluate_context.py \
  --questions data/processed/validation_retrieval.parquet \
  --documents data/processed/legal_documents_categorized.csv \
  --limit 100
```

ผลจริงจากชุดตรวจสอบจำนวน 100 คำถาม บันทึกไว้ใน
`results/context_metrics.csv`:

```text
ContextRecall@1:     0.6900
ContextRecall@3:     0.8000
ContextRecall@5:     0.8300
ContextPrecision@5:  0.2060
ContextMRR@5:        0.749833
```

## 7. การประเมินคำตอบจาก Gemini

สร้างพรอมต์ที่มีหลักฐานอ้างอิงจากหลักฐาน 5 อันดับแรกสุดท้าย โดยสร้างคำตอบด้วย:

```text
Model: gemini-3.5-flash-lite
Temperature: 0.0
```

ชุดตรวจสอบ 100 คำถาม โดยคำนวณจาก
`results/qa_predictions_validation_100.jsonl` และบันทึกใน
`results/answer_metrics_validation_100.csv`:

```text
F1 ของโทเคนคำตอบ:       0.395647
ROUGE-1 (F1):            0.395647
ROUGE-2 (F1):            0.281292
ROUGE-L (F1):            0.302869
การอ้างอิงถูกต้อง:        0.640000
งดตอบ:                   0.280000
```

ตัวอย่างชุดทดสอบที่ไม่ได้นำไปปรับแต่ง จำนวน 100 คำถาม:

```text
F1 ของโทเคนคำตอบ:       0.487709
ROUGE-1 (F1):            0.487709
ROUGE-2 (F1):            0.390759
ROUGE-L (F1):            0.371400
การอ้างอิงถูกต้อง:        0.800000
งดตอบ:                   0.130000
```

คำสั่งสร้างและประเมินผล:

```bash
python scripts/export_qa_prompts.py \
  --questions data/processed/validation_retrieval.parquet \
  --limit 100 \
  --output results/qa_prompts_validation_100.jsonl

python scripts/generate_gemini_answers.py \
  --input results/qa_prompts_validation_100.jsonl \
  --output results/qa_predictions_validation_100.jsonl \
  --model gemini-3.5-flash-lite \
  --retry-errors

python scripts/evaluate_answers.py \
  --predictions results/qa_predictions_validation_100.jsonl \
  --references data/processed/validation_retrieval.parquet \
  --output results/answer_metrics_validation_100.csv
```

ไฟล์คำตอบ validation และ test จำนวน 100 คำถามสร้างสำเร็จครบชุดละ 100/100 รายการ
โดยไม่มีข้อผิดพลาด ตัวเลขทั้งสองชุดสร้างด้วย API key ที่ป้อนแบบซ่อนค่าและไม่ได้
บันทึกลง repository ตัวเลขชุดหลักใน
`results/answer_metrics.csv` มาจากคำตอบที่สำเร็จเพียง 13 จาก 100 รายการ และมี
ค่า F1 เฉลี่ย 0.032906, ROUGE-1 0.032906, ROUGE-2 0.021367 และ ROUGE-L 0.027276
เนื่องจากอีก 87 รายการเป็นข้อผิดพลาดจากโควต้าเดิม

เครื่องมือสร้างคำตอบรองรับการลองใหม่พร้อมหน่วงเวลา การทำงานต่อจากรายการเดิม
และการบันทึกข้อผิดพลาด โดยอ่าน API key จาก `GEMINI_API_KEY` เท่านั้น
และไม่จัดเก็บไว้ใน repository

## 8. การประเมินโดยมนุษย์

ผู้ประเมินสามารถให้คะแนนคำตอบที่สร้างขึ้นจำนวน 50-100 รายการ ตั้งแต่ 0 ถึง 2:

| เกณฑ์ | 0 | 1 | 2 |
|---|---|---|---|
| ความถูกต้องของคำตอบ | ผิด | ถูกบางส่วน | ถูกต้อง |
| ความสอดคล้องกับหลักฐาน | ไม่มีหลักฐานรองรับ | รองรับบางส่วน | รองรับทั้งหมด |
| ความถูกต้องของการอ้างอิง | ผิดหรือไม่มีการอ้างอิง | ถูกบางส่วน | กฎหมายและมาตราถูกต้อง |
| ความครบถ้วน | ไม่ครบถ้วน | ครบถ้วนบางส่วน | ครบถ้วน |
| พฤติกรรมการงดตอบ | แต่งข้อมูล | ไม่ชัดเจน | ปฏิเสธหรือให้เงื่อนไขอย่างถูกต้อง |

คอลัมน์ที่แนะนำ:

```text
question, answer, retrieved_sources, answer_correctness,
evidence_faithfulness, citation_correctness, completeness,
abstention_behavior, notes, reviewer_id
```

## 9. เดโมและฮาร์ดแวร์

เดโมบนบรรทัดคำสั่ง:

```bash
python scripts/run_qa_demo.py
```

เดโมบนเว็บ:

```bash
python scripts/run_web_demo.py \
  --documents data/processed/legal_documents_categorized.csv
```

ฮาร์ดแวร์:

```bash
python scripts/check_hardware.py
```

สภาพแวดล้อมปัจจุบันไม่มี NVIDIA CUDA GPU จึงใช้ CPU เป็นทางเลือกสำรอง

## 10. ข้อจำกัด

- ป้ายกำกับหมวดหมู่สร้างจากกฎ ไม่ใช่ป้ายกำกับอ้างอิง
- การประเมินคำตอบใช้ตัวอย่างชุดตรวจสอบและชุดทดสอบอย่างละ 100 คำถาม
- คำถามภาษาพูดบางส่วนยังค้นพบหลักฐานที่ไม่ชัดเจน แม้ค่าเมตริกรวมจะดี
  ระบบควรงดตอบเมื่อหลักฐานไม่เกี่ยวข้องอย่างชัดเจน
- ยังแนะนำให้มีผู้เชี่ยวชาญด้านกฎหมายประเมิน เพื่อสนับสนุนข้อสรุปด้านระบบถามตอบกฎหมายให้หนักแน่นขึ้น

## 11. โครงสร้างรายงาน

1. นิยามปัญหา
2. ชุดข้อมูลและงานที่เกี่ยวข้อง
3. วิธีการค้นหาและจัดอันดับซ้ำ
4. ระเบียบวิธีชุดตรวจสอบและชุดทดสอบ
5. ผลการค้นหาและผลคำตอบ
6. การค้นหาที่กรองตามหมวดหมู่
7. การวิเคราะห์ข้อผิดพลาดและข้อจำกัด
8. บทสรุป
