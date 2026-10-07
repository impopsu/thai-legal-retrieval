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

เตรียมตัวอย่างสำหรับตรวจหมวดด้วยคนไว้ใน
`results/category_accuracy_check.csv` โดยสุ่มไม่เกิน 30 มาตราต่อ auto-category
และใส่ครบทุกมาตราในหมวดที่มีน้อยกว่า ช่อง `human_category` และ `correct`
ยังว่างเพื่อรอผู้ตรวจ จึงยังไม่มีค่า accuracy ที่ยืนยันโดยมนุษย์
ดูขั้นตอนและวิธีสรุป accuracy ได้ใน [คู่มือตรวจหมวดกฎหมาย](category-review-guide.md)

ผลของการกรองหมวดต่อ Hybrid MiniLM + reranker บน validation:

| Setting | Recall@5 | คำถามที่ใช้ | ข้ามเพราะไม่มีเฉลย |
|---|---:|---:|---:|
| ไม่กรองหมวด | 0.749158 | 99 | 1 |
| กรองด้วยหมวด oracle | 0.814815 | 99 | 1 |

ผลจริงอยู่ใน `results/category_filter_recall.csv` ใช้ validation 100 คำถามเดียวกัน
และกรองผู้สมัคร 20 รายการก่อน rerank เหลือ 5 รายการ หมวด oracle กำหนดจาก
หมวดเสียงข้างมากของ positive contexts ที่ map เข้าเอกสารได้; หมวดดังกล่าวอิง
auto-classifier ไม่ใช่ human-verified label ดังนั้นผลนี้เป็นการจำลองกรณีเลือกหมวด
ถูก ไม่ใช่ผลความแม่นยำของ classifier

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

ผลจากการประเมิน 100 คำถามต่อชุด:

| Split | Configuration | Recall@1 | Recall@3 | Recall@5 | Precision@5 | MRR@5 |
|---|---|---:|---:|---:|---:|---:|
| Validation | Hybrid, no reranker | 0.5600 | 0.7200 | 0.7500 | 0.1840 | 0.635333 |
| Validation | Hybrid + reranker | 0.6900 | 0.8000 | 0.8300 | 0.2060 | 0.749833 |
| Test | Hybrid, no reranker | 0.6900 | 0.8100 | 0.8600 | 0.1720 | 0.754833 |
| Test | Hybrid + reranker | 0.7300 | 0.8600 | 0.8700 | 0.1740 | 0.787500 |

ผล validation แบบ no-reranker อยู่ใน `results/context_metrics_no_reranker.csv`;
ผล validation แบบ reranker อยู่ใน `results/context_metrics.csv`;
ผล test อยู่ใน `results/context_metrics_test_no_reranker.csv` และ
`results/context_metrics_test.csv` ตาม configuration ในตาราง

รัน validation แบบไม่มี reranker:

```bash
python scripts/evaluate_context.py --questions data/processed/validation_retrieval.parquet \
  --limit 100 --no-reranker --output results/context_metrics_no_reranker.csv
```

รัน test แบบมี reranker:

```bash
python scripts/evaluate_context.py --questions data/raw/test-00000-of-00001.parquet \
  --limit 100 --output results/context_metrics_test.csv
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

จากการประเมินจริงบนไฟล์ที่มีอยู่แล้ว `results/qa_predictions_validation_100.jsonl`
มี 200 บรรทัดทั้งหมด ประกอบด้วย 100 บรรทัดที่มี `status=error` และ 100 บรรทัดที่
ประสบความสำเร็จ แล้วจึงถูกประเมินต่อ. หลัง filter รายการที่ไม่สำเร็จออกแล้ว
script `scripts/evaluate_answers.py` ประเมินได้ 100 คำถาม โดยบันทึกผลลง
`results/answer_metrics_validation_100.csv` พร้อมตัวชี้วัด:

```text
answer_token_f1: 0.395647
rouge1:           0.395647
rouge2:           0.281292
rougeL:           0.302869
citation_correct: 0.640000
abstained:        0.280000
```

สำหรับชุดทดสอบ `results/qa_predictions_test_100.jsonl` มี 100 บรรทัดทั้งหมดและ
ประเมินสำเร็จครบ 100/100 รายการ โดยผลจริงใน `results/answer_metrics_test_100.csv`
คือ:

```text
answer_token_f1: 0.487709
rouge1:           0.487709
rouge2:           0.390759
rougeL:           0.371400
citation_correct: 0.800000
abstained:        0.130000
```

ประวัติการเรียก model ที่ใช้จริงคือมีการลองใช้ model เดิม `gemini-2.5-flash-lite`
ซึ่ง Google ปิดใช้งานสำหรับผู้ใช้ใหม่ ทำให้ได้รับ HTTP 404 จากชื่อ model เดิม
จากนั้นจึง retry ชุด validation ด้วย `gemini-3.5-flash-lite` และได้ผลสำเร็จครบ
100/100 ตาม 100 บรรทัด success ในไฟล์ข้างต้น. ไฟล์คำตอบและ metrics ที่ระบุในหัวข้อนี้
ถูกเก็บไว้ใน `results/` และ tracked ใน repository แล้ว. เครื่องมือสร้างคำตอบอ่านค่า
`GEMINI_API_KEY` เท่านั้น และไม่จัดเก็บค่า API ลง repository

## 8. การประเมินโดยมนุษย์

เตรียมตัวอย่างจากคำตอบจริงไว้ใน `results/human_eval_answers_rubric.jsonl` จำนวน
40 รายการ แบ่งจาก validation และ test ชุดละ 20 รายการ แต่ละรายการมีคำถาม
คำตอบที่สร้าง คำตอบอ้างอิง positive contexts และ retrieved sources พร้อมช่องคะแนน
ที่เว้นว่างไว้ให้ผู้ประเมินกรอก จึงยังไม่มีผลคะแนน human evaluation ในขณะนี้

```bash
python scripts/prepare_human_eval.py --sample-per-split 20 --seed 42
```

ให้คะแนน 1-5 ในห้ามิติ: correctness, faithfulness, citation, completeness และ
abstention ตาม [คู่มือประเมินคำตอบ](human-evaluation-guide.md) ซึ่งอธิบายเกณฑ์
คะแนนและการพิจารณาการงดตอบ

## 9. Runtime และหน่วยความจำ

วัดบน validation ชุดเดียวกัน ค่าเฉลี่ย latency นับเฉพาะการค้นหาแต่ละคำถาม
ส่วน peak RSS รวมการโหลดเอกสาร โมเดล และ warm-up วัดแยก process ต่อวิธี:

| Method | Avg latency (ms/question) | Peak RSS (MB) | Questions |
|---|---:|---:|---:|
| BM25 | 26.214 | 1118.723 | 100 |
| MiniLM | 13.103 | 2167.488 | 100 |
| Hybrid MiniLM | 32.826 | 2227.152 | 100 |
| BGE-M3 | 25.913 | 3944.535 | 100 |
| Hybrid MiniLM + reranker | 142.094 | 2398.039 | 100 |

ผลจริงอยู่ใน `results/runtime_benchmark.csv` วัดบน validation 100 คำถามครบทั้งห้าวิธี
โดยนำผล BGE-M3 มารันบน Colab เนื่องจากการสร้าง embeddings บน CPU ในเครื่องนี้ใช้เวลานานเกินช่วงเวลาที่รันได้

Hybrid MiniLM + reranker ยังคงเป็น configuration หลัก เพราะเลือก encoder MiniLM
ซึ่งมี peak RSS ต่ำกว่า BGE-M3 ใน benchmark นี้ ขณะที่ reranker เพิ่มคุณภาพการจัดอันดับ
ตามผล retrieval ข้างต้น แม้ BGE-M3 จะได้ Recall@5 สูงกว่าในบางการเปรียบเทียบ
แต่ใช้หน่วยความจำสูงกว่า; Hybrid+Reranker จึงเป็นจุดสมดุลด้านคุณภาพกับหน่วยความจำ
ทั้งนี้มี latency สูงกว่า BGE-M3 ที่ไม่ใช้ reranker ตามผลในตาราง

รัน benchmark ครบทุกวิธี:

```bash
python scripts/benchmark_runtime.py --limit 100 \
  --methods bm25 minilm hybrid bge_m3 hybrid_reranker
```

## 10. เดโมและฮาร์ดแวร์

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

## 11. ข้อจำกัด

- ป้ายกำกับหมวดหมู่สร้างจากกฎ ไม่ใช่ป้ายกำกับอ้างอิง
- การประเมินคำตอบใช้ตัวอย่างชุดตรวจสอบและชุดทดสอบอย่างละ 100 คำถาม
- คำถามภาษาพูดบางส่วนยังค้นพบหลักฐานที่ไม่ชัดเจน แม้ค่าเมตริกรวมจะดี
  ระบบควรงดตอบเมื่อหลักฐานไม่เกี่ยวข้องอย่างชัดเจน
- ยังแนะนำให้มีผู้เชี่ยวชาญด้านกฎหมายประเมิน เพื่อสนับสนุนข้อสรุปด้านระบบถามตอบกฎหมายให้หนักแน่นขึ้น

## 12. โครงสร้างรายงาน

1. นิยามปัญหา
2. ชุดข้อมูลและงานที่เกี่ยวข้อง
3. วิธีการค้นหาและจัดอันดับซ้ำ
4. ระเบียบวิธีชุดตรวจสอบและชุดทดสอบ
5. ผลการค้นหาและผลคำตอบ
6. การค้นหาที่กรองตามหมวดหมู่
7. การวิเคราะห์ข้อผิดพลาดและข้อจำกัด
8. บทสรุป
