# สรุปงานที่เพิ่มในโครงงาน Thai Legal QA / Retrieval

เอกสารนี้รวบรวมการแก้ไขและการประเมินเพิ่มเติม เพื่อใช้รายงานความคืบหน้ากับอาจารย์ ตัวเลขในส่วนผลการทดลองอ้างจากไฟล์ CSV ใน `results/` โดยตรง งานที่เป็น AI draft แยกจากผลที่ต้องให้ผู้ประเมินมนุษย์ยืนยัน

## 1. สรุปภาพรวม

งานเพิ่มเติมครอบคลุมการทำให้ retrieval evaluation ทนต่อข้อมูลเฉลยที่หาย การจัดการ prediction ที่จับคู่ reference ไม่ได้ การแก้ `.gitignore` ของผลลัพธ์ การเปรียบเทียบ context retrieval การวัด runtime/หน่วยความจำ การเตรียม human-review templates การทดลอง category filtering และการเพิ่ม retrieval F1

งานที่รันและมีไฟล์ผลรองรับแล้ว:

- Runtime และ peak RSS ของ BM25, MiniLM, Hybrid MiniLM, BGE-M3 และ Hybrid MiniLM + reranker
- Context metrics บน validation/test ทั้งแบบมีและไม่มี reranker
- Retrieval precision, recall และ F1@k บน test subset
- Category-filter Recall@5 แบบไม่กรองเทียบกับ oracle-category filter
- AI review drafts สำหรับคำตอบและข้อเสนอหมวดหมู่ ซึ่งยังไม่ใช่ผลประเมินจากมนุษย์

## 2. การแก้ความทนทานของ evaluation

### A1–A2: Missing ground truth และ missing document

`legal_qa/evaluation.py` ข้าม positive context ที่อ้าง `unique_key` ซึ่งไม่มีในคลังเอกสาร แทนการหยุดด้วย `KeyError` พร้อมนับจำนวน missing documents และคำถามที่ไม่มีเอกสารเฉลยในคลังแยกไว้

คำถามที่ไม่มี `positive_contexts` จะไม่ถูกนับเป็นผลค้นหาพลาดและไม่เข้า denominator ของ Recall/MRR ตัว metrics รายงาน `num_questions` ที่ใช้คำนวณจริง และ `skipped_no_ground_truth` แยกจากจำนวนคำถามทั้งหมด

### A3: Robust answer evaluation

`scripts/evaluate_answers.py` ข้ามและ log รายการที่ JSON เสียรูป, ไม่มี question/answer, จับคู่ reference ไม่ได้, มี status ที่ไม่สำเร็จ หรือเกิด evaluation error แทนการทำให้ทั้งสคริปต์หยุด รวมทั้งยังสร้าง CSV header ได้เมื่อไม่มีคำตอบที่ประเมินได้

### A4–A5: เอกสารและ Git ignore

หัวข้อ Gemini ใน `docs/project-summary.md` แก้ให้ตรงกับประวัติจริง: การเรียก `gemini-2.5-flash-lite` ครั้งแรกได้ HTTP 404 แล้ว retry ชุด validation ด้วย `gemini-3.5-flash-lite` จนมีคำตอบสำเร็จครบตามไฟล์ผล ส่วนไฟล์ answer metrics/predictions ที่กล่าวถึง tracked ใน repository แล้ว

`.gitignore` เปิดให้ผลลัพธ์ CSV, JSONL และ PT ใน `results/` ไม่ถูกซ่อนอยู่ใน ignored list ตรวจ `git status --ignored` หลังสร้างผลลัพธ์แล้ว

## 3. ผล Retrieval F1

เพิ่ม `scripts/evaluate_retrieval_f1.py` และผล `results/retrieval_f1_test_100.csv` ซึ่งคำนวณ Precision, Recall และ F1 แบบเฉลี่ยรายคำถามที่ k=1, 3 และ 5 สำหรับคำถาม 100 ข้อแรกของ test split คำถามทั้ง 100 ข้อมี positive documents ที่ map เข้า corpus ได้

| Method | F1@1 | F1@3 | F1@5 |
|---|---:|---:|---:|
| BM25 | 0.7000 | 0.4150 | 0.2900 |
| Semantic MiniLM | 0.3900 | 0.2700 | 0.1900 |
| Hybrid MiniLM | 0.6900 | 0.4050 | 0.2867 |
| Hybrid MiniLM + Cross-Encoder Reranker | 0.7300 | 0.4300 | 0.2900 |

รันซ้ำได้ด้วย:

```bash
python scripts/evaluate_retrieval_f1.py --limit 100 \
  --methods bm25 minilm hybrid hybrid_reranker \
  --output results/retrieval_f1_test_100.csv
```

F1 นี้เป็น **retrieval F1** ไม่ใช่ answer Token F1 หรือ ROUGE ของคำตอบที่สร้าง

## 4. Context evaluation

`scripts/evaluate_context.py` รันเพิ่มบน validation แบบ no-reranker และ test ทั้งแบบ no-reranker/with-reranker เพื่อเทียบกับ validation reranker result ที่มีอยู่แล้ว ผลแต่ละ configuration ใช้ 100 คำถาม

| Split | Configuration | Recall@1 | Recall@3 | Recall@5 | Precision@5 | MRR@5 |
|---|---|---:|---:|---:|---:|---:|
| Validation | Hybrid, no reranker | 0.5600 | 0.7200 | 0.7500 | 0.1840 | 0.635333 |
| Validation | Hybrid + reranker | 0.6900 | 0.8000 | 0.8300 | 0.2060 | 0.749833 |
| Test | Hybrid, no reranker | 0.6900 | 0.8100 | 0.8600 | 0.1720 | 0.754833 |
| Test | Hybrid + reranker | 0.7300 | 0.8600 | 0.8700 | 0.1740 | 0.787500 |

ผลอยู่ใน `results/context_metrics.csv`, `results/context_metrics_no_reranker.csv`, `results/context_metrics_test.csv` และ `results/context_metrics_test_no_reranker.csv`

## 5. Runtime และ memory benchmark

เพิ่ม `scripts/benchmark_runtime.py` วัด latency เฉลี่ยต่อคำถามและ peak resident memory (RSS) แยก process ต่อวิธี ใช้ validation 100 คำถามชุดเดียวกัน โดย latency ไม่นับ model load แต่ peak RSS รวมการโหลดข้อมูล/โมเดลและ warm-up

| Method | Avg latency (ms/question) | Peak RSS (MB) | Questions |
|---|---:|---:|---:|
| BM25 | 24.818 | 1120.273 | 100 |
| MiniLM | 12.706 | 2144.418 | 100 |
| Hybrid MiniLM | 30.207 | 2229.637 | 100 |
| BGE-M3 | 23.026 | 3619.578 | 100 |
| Hybrid MiniLM + reranker | 136.590 | 2399.734 | 100 |

ผลจริงอยู่ใน `results/runtime_benchmark.csv` โดย benchmark รอบล่าสุดรันทั้งห้าวิธีใน Colab environment เดียวกัน
ใช้ validation 100 คำถามชุดเดียวกัน และใช้วิธีจับเวลาเดียวกัน เพื่อให้การเปรียบเทียบ latency และ peak RSS อยู่ภายใต้สภาพแวดล้อมเดียวกัน
ตัวเลขจึงใช้เป็น benchmark ของ configuration ใน environment นี้ ไม่ได้หมายความว่าเป็นการเปรียบเทียบประสิทธิภาพฮาร์ดแวร์คนละเครื่อง

Hybrid MiniLM + reranker ถูกเลือกเป็น configuration หลักจากสมดุลด้าน retrieval quality และ memory: ผล retrieval เดิมของ reranker ดีขึ้นจาก Hybrid baseline และ runtime RSS ต่ำกว่า BGE-M3 ใน benchmark นี้ อย่างไรก็ตาม BGE-M3 มี latency ต่ำกว่า Hybrid+Reranker ใน runtime table และ Recall@5 สูงกว่าใน retrieval comparison บางชุด จึงเป็น trade-off ไม่ใช่ผู้ชนะทุก metric

## 6. ผล category filtering

เพิ่มการเปรียบเทียบ Hybrid MiniLM + Cross-Encoder Reranker แบบไม่กรองกับแบบกรองด้วย oracle category บน validation ชุดเดียวกัน 100 คำถาม โดยใช้ positive contexts ใน corpus เลือกหมวดเสียงข้างมากเป็น oracle และวัด Recall@5 เฉพาะ positive documents ในหมวดนั้น มีหนึ่งคำถามไม่มี ground truth จึงเหลือ 99 คำถามใน denominator

| Setting | Recall@5 | Questions used | Skipped: no ground truth |
|---|---:|---:|---:|
| Unfiltered | 0.749158 | 99 | 1 |
| Oracle category filter | 0.814815 | 99 | 1 |

ไฟล์ผลคือ `results/category_filter_recall.csv` หมวด oracle สร้างจาก auto-category ของ positive documents ไม่ใช่ human-verified label ผลนี้จึงจำลองกรณีผู้ใช้เลือกหมวดได้ถูก ไม่ได้วัดความแม่นยำของตัวจำแนกหมวด

## 7. Human evaluation และ AI drafts

เตรียม `results/human_eval_answers_rubric.jsonl` เป็นแบบฟอร์ม 40 คำตอบ แบ่ง validation/test อย่างละ 20 ข้อ มีช่องให้ผู้ประเมินกรอก correctness, faithfulness, citation, completeness และ abstention ระดับ 1–5 คู่มืออยู่ใน `docs/human-evaluation-guide.md`

เพื่อช่วยผู้ประเมิน มี `results/human_eval_ai_review_draft.jsonl` ซึ่งใส่คะแนนเบื้องต้นและเหตุผลโดย AI ไว้ใน field แยก และ `results/category_ai_review_draft.csv` ซึ่งมีข้อเสนอหมวดจาก rule classifier บนชื่อกฎหมาย/เนื้อหามาตรา

ไฟล์ AI เหล่านี้เป็นเพียง draft ไม่ใช่ human evaluation หรือ category accuracy ที่ยืนยันแล้ว ช่อง human scores/category ยังคงว่าง ต้องให้ผู้ประเมินอ่านหลักฐาน ตรวจ แก้ไข และยืนยันก่อนใช้เป็นผลสุดท้าย คู่มือ category review อยู่ใน `docs/category-review-guide.md`

## 8. Answer F1 ที่มีอยู่ก่อน

นอกจาก retrieval F1 ในหัวข้อ 3 ยังมี answer Token F1 และ ROUGE จากคำตอบ Gemini 100 ข้อต่อ split อยู่ก่อนแล้ว:

| Split | Answer Token F1 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|
| Validation | 0.395647 | 0.395647 | 0.281292 | 0.302869 |
| Test | 0.487709 | 0.487709 | 0.390759 | 0.371400 |

ค่าดังกล่าวอยู่ใน `results/answer_metrics_validation_100.csv` และ `results/answer_metrics_test_100.csv` ไม่ควรสับสนกับ retrieval F1@k

## 9. ข้อจำกัดและงานต่อ

- ผล retrieval F1 ใหม่เป็น test subset 100 คำถาม ไม่ใช่ test split ทั้งหมด
- Context evaluation และ category-filter experiment ใช้ชุดตัวอย่าง 100 คำถามต่อ configuration
- Runtime CSV ไม่มีข้อมูล hardware identifier จึงควรระบุสภาพแวดล้อม Colab เพิ่ม หากนำไปเทียบกับเครื่องอื่น
- คะแนน B3 และหมวด B4 ยังเป็น AI-assisted drafts; ต้องมีมนุษย์ตรวจและยืนยันก่อนเรียกว่า human evaluation/category accuracy
- Runtime F1 CSV และ scripts ถูกเก็บใน Git ตาม commits ของงาน; ยังมี staged embedding cache local ซึ่งแยกจากผลสรุปและไม่จำเป็นต่อการอธิบายตัวเลข

## 10. Commit trail

งานถูก commit/push แยกตามส่วนเพื่อให้ตรวจสอบย้อนหลังได้. Commit ที่เกี่ยวกับชุดงานนี้ ได้แก่:

```text
3ea0539 Harden answer evaluation and result tracking
d5a16c4 Add paired context evaluation results
84f3fa2 Prepare human answer evaluation rubric
1687c26 Prepare human legal category review sample
03cc68e Benchmark oracle category filtering
c523142 Complete runtime benchmark with Colab BGE results
91690da Resolve runtime benchmark paths from repository root
57e4a23 Explain missing Git LFS benchmark data
a914be3 Add provisional AI review drafts
b626a61 Add test retrieval F1 metrics
```

## 11. สคริปต์และไฟล์ผลที่ใช้ซ้ำ

```text
scripts/benchmark_runtime.py
scripts/evaluate_context.py
scripts/evaluate_category_checks.py
scripts/category_checks.py
scripts/evaluate_retrieval_f1.py
scripts/prepare_human_eval.py
scripts/prepare_ai_review_drafts.py
results/runtime_benchmark.csv
results/retrieval_f1_test_100.csv
results/category_filter_recall.csv
results/human_eval_answers_rubric.jsonl
results/human_eval_ai_review_draft.jsonl
results/category_accuracy_check.csv
results/category_ai_review_draft.csv
```
