# สรุปค่าประเมินและที่มาของผล

เอกสารนี้รวบรวมตัวเลขที่มีอยู่ใน `results/` พร้อมอธิบายว่าคำนวณจากอะไร ใช้ชุดใด และเปรียบเทียบกับอะไร อัปเดตตาม artifacts ที่อยู่ใน repository ณ วันที่ 2026-10-08 ตัวเลขในตารางแสดงเป็นสัดส่วน 0–1 เว้นแต่ระบุหน่วยไว้ชัดเจน

## ภาพรวมข้อมูล

| รายการ | จำนวน | ที่มา/ข้อควรทราบ |
|---|---:|---|
| Train questions | 8,211 | จำนวนที่รายงานใน [สรุปโครงงาน](project-summary.md) |
| Validation questions | 1,643 | ใช้เลือกค่าและ configuration; `data/processed/validation_retrieval.parquet` |
| Test questions | 3,742 | Official test split; ใช้รายงานผลสุดท้าย ไม่ใช้ tune |
| Legal sections ใน corpus | 4,545 จาก 35 กฎหมาย | จำนวนที่รายงานใน [สรุปโครงงานสำหรับอาจารย์](advisor-summary.md) |
| Final retrieval candidates / evidence | 20 / 5 | Hybrid MiniLM, `alpha=0.5`, แล้ว rerank ด้วย `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |

## นิยามการวัด

- Retrieval `Recall@k`: สัดส่วนคำถามที่พบเอกสารเฉลยอย่างน้อยหนึ่งรายการใน top-k; ผลหลักคำนวณจาก `positive_contexts` และระบุจำนวนคำถามที่ถูกข้ามไว้ใน CSV
- `MRR@5`: ค่าเฉลี่ย reciprocal rank ของเอกสารเฉลยตัวแรกในห้าอันดับแรก; ไม่พบให้ค่า 0
- Retrieval `Precision@k`, `Recall@k`, `F1@k`: คำนวณต่อคำถามจากเซตเอกสารที่ดึงได้เทียบ `positive_contexts` แล้วเฉลี่ยรายคำถาม (macro average)
- Context metrics: ใช้ `unique_key` เทียบผลค้นหากับ `positive_contexts`; `ContextPrecision@5` นับ hit ใน top 5 หารด้วย 5 ต่อคำถาม และเฉลี่ย ส่วน Recall@k เป็น hit/miss ต่อคำถาม
- Answer Token F1 และ ROUGE: ตัดคำภาษาไทยด้วย PyThaiNLP `newmm` แล้วคำนวณเทียบ `positive_answer`; เป็นความคล้ายเชิงข้อความ ไม่ใช่การตัดสินความถูกต้องทางกฎหมาย
- `citation_correct`: heuristic ตรวจว่าชื่อกฎหมายและเลขมาตราจาก positive context ปรากฏในคำตอบ; ไม่ใช่การตรวจความถูกต้องทางกฎหมายโดยผู้เชี่ยวชาญ
- `abstained`: ตรวจว่าคำตอบมีวลี `ไม่พบข้อมูลเพียงพอจากเอกสารที่ค้นได้` หรือไม่

สูตรและ protocol อยู่ใน [retrieval evaluation](../legal_qa/evaluation.py), [context evaluation](../legal_qa/context_evaluation.py), [answer evaluation](../legal_qa/answer_evaluation.py) และ scripts ที่ลิงก์ในแต่ละหัวข้อ

## Retrieval: Validation เต็มชุด

เทียบ retriever ด้วย `positive_contexts` ของ validation 1,643 คำถาม ค่า `alpha=0.5` ใช้กับ Hybrid; BGE-M3 สองแถวเป็น comparison ไม่ใช่ final pipeline

| วิธี | Recall@1 | Recall@3 | Recall@5 | MRR@5 | แหล่งผล |
|---|---:|---:|---:|---:|---|
| BM25 | 0.494827 | 0.651856 | 0.715764 | 0.578271 | [retrieval_benchmark_validation_bm25_minilm.csv](../results/retrieval_benchmark_validation_bm25_minilm.csv) |
| Semantic MiniLM | 0.321363 | 0.472915 | 0.550822 | 0.406147 | [retrieval_benchmark_validation_bm25_minilm.csv](../results/retrieval_benchmark_validation_bm25_minilm.csv) |
| Hybrid MiniLM, alpha=0.5 | 0.567255 | 0.720633 | 0.766890 | 0.646186 | [validation_alpha_results.csv](../results/validation_alpha_results.csv), [reranker_validation_results.csv](../results/reranker_validation_results.csv) |
| BGE-M3 | 0.654291 | 0.811321 | 0.867316 | 0.737929 | [retrieval_benchmark.csv](../results/retrieval_benchmark.csv) |
| Hybrid BGE-M3, alpha=0.5 | 0.640901 | 0.781497 | 0.841144 | 0.716606 | [retrieval_benchmark.csv](../results/retrieval_benchmark.csv) |

วิธีคำนวณ/เปรียบเทียบ: [benchmark_retrievers.py](../scripts/benchmark_retrievers.py) และ [evaluate_retrieval.py](../scripts/evaluate_retrieval.py); จำนวนคำถามใน CSV ยืนยันว่า validation เต็มชุดมี 1,643 รายการ

## เลือก Alpha

เปรียบเทียบ Hybrid MiniLM บน validation เดียวกันเพื่อเลือกน้ำหนัก BM25 ใน `hybrid_score = alpha * normalized_bm25 + (1 - alpha) * normalized_semantic` ค่า 0.5 ได้ Recall และ MRR สูงสุดในสามค่าที่ทดลอง จึงนำไปใช้ใน final configuration

| Alpha | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---:|---:|---:|---:|---:|
| 0.25 | 0.537432 | 0.699939 | 0.753500 | 0.620491 |
| 0.50 | 0.567255 | 0.720633 | 0.766890 | 0.646186 |
| 0.75 | 0.530736 | 0.688984 | 0.747413 | 0.614303 |

ที่มา: [validation_alpha_results.csv](../results/validation_alpha_results.csv); วิธีรันอยู่ใน [select_alpha.py](../scripts/select_alpha.py)

## ผล Cross-Encoder Reranker

เทียบ Hybrid MiniLM baseline กับ reranker แบบ paired บนชุดคำถามเดียวกัน โดยดึง candidate top-20 แล้วจัดอันดับใหม่เป็น top-5 ด้วย `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`

| Split | จำนวน | Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---|---:|---|---:|---:|---:|---:|
| Validation | 1,643 | Hybrid baseline | 0.567255 | 0.720633 | 0.766890 | 0.646186 |
| Validation | 1,643 | Hybrid + reranker | 0.699939 | 0.822276 | 0.849665 | 0.761828 |
| Test | 3,742 | Hybrid baseline | 0.588188 | 0.743453 | 0.799038 | 0.670056 |
| Test | 3,742 | Hybrid + reranker | 0.719401 | 0.835382 | 0.862640 | 0.778986 |

Validation อยู่ใน [reranker_validation_results.csv](../results/reranker_validation_results.csv); test อยู่ใน [final_reranker_test_results.csv](../results/final_reranker_test_results.csv) และมีผล baseline แยกใน [final_test_results.csv](../results/final_test_results.csv). รันด้วย [evaluate_reranker.py](../scripts/evaluate_reranker.py). Test ใช้รายงานผลหลังเลือก configuration แล้ว ไม่ใช่ชุดสำหรับเลือกโมเดล

ผล test ของ retriever พื้นฐานจาก [final_test_results.csv](../results/final_test_results.csv):

| วิธี | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| BM25 | 0.555852 | 0.701229 | 0.752806 | 0.632821 |
| Semantic MiniLM | 0.335382 | 0.486104 | 0.551309 | 0.416818 |
| Hybrid MiniLM, alpha=0.5 | 0.588188 | 0.743453 | 0.799038 | 0.670056 |

## Retrieval F1: 100 คำถามแรกของ Test

วัด Precision, Recall และ F1 รายคำถามที่ k=1, 3, 5 แล้วเฉลี่ย; ทั้ง 100 คำถามมี positive document ใน corpus จึงไม่มีรายการถูกข้าม ตารางนี้เป็น test subset ไม่ใช่ผล test เต็มชุด

| วิธี | P/R/F1@1 | P/R/F1@3 | P/R/F1@5 | MRR@5 |
|---|---|---|---|---:|
| BM25 | 0.700 / 0.700 / 0.700 | 0.277 / 0.830 / 0.415 | 0.174 / 0.870 / 0.290 | 0.758500 |
| Semantic MiniLM | 0.390 / 0.390 / 0.390 | 0.180 / 0.540 / 0.270 | 0.114 / 0.570 / 0.190 | 0.467000 |
| Hybrid MiniLM | 0.690 / 0.690 / 0.690 | 0.270 / 0.810 / 0.405 | 0.172 / 0.860 / 0.286667 | 0.754833 |
| Hybrid MiniLM + reranker | 0.730 / 0.730 / 0.730 | 0.287 / 0.860 / 0.430 | 0.174 / 0.870 / 0.290 | 0.787500 |

ที่มา: [retrieval_f1_test_100.csv](../results/retrieval_f1_test_100.csv), คำนวณด้วย [evaluate_retrieval_f1.py](../scripts/evaluate_retrieval_f1.py). อย่าสับสน Retrieval F1 กับ Answer Token F1 ด้านล่าง

## Context Metrics: 100 คำถามต่อชุด

เปรียบเทียบผล top-5 กับ positive contexts สำหรับ 100 คำถามแรกของแต่ละ split; Hybrid MiniLM alpha=0.5 ใช้ทั้งสองแบบ, reranker รับ top-20 และคืน top-5

| Split | Reranker | Recall@1 | Recall@3 | Recall@5 | Precision@5 | MRR@5 |
|---|---|---:|---:|---:|---:|---:|
| Validation | ไม่ใช้ | 0.5600 | 0.7200 | 0.7500 | 0.1840 | 0.635333 |
| Validation | ใช้ | 0.6900 | 0.8000 | 0.8300 | 0.2060 | 0.749833 |
| Test | ไม่ใช้ | 0.6900 | 0.8100 | 0.8600 | 0.1720 | 0.754833 |
| Test | ใช้ | 0.7300 | 0.8600 | 0.8700 | 0.1740 | 0.787500 |

แหล่งผล: [context_metrics_no_reranker.csv](../results/context_metrics_no_reranker.csv), [context_metrics.csv](../results/context_metrics.csv), [context_metrics_test_no_reranker.csv](../results/context_metrics_test_no_reranker.csv), [context_metrics_test.csv](../results/context_metrics_test.csv). รันด้วย [evaluate_context.py](../scripts/evaluate_context.py). ตัวเลขเหล่านี้เป็น 100-query context evaluation ไม่ใช่ full-split retrieval benchmark ด้านบน และ `ContextPrecision@5` เป็น precision ของ 5 ช่องผลลัพธ์ ไม่ใช่เอกสารที่ไม่ซ้ำทั้ง corpus

## Answer Metrics: Gemini

สร้างคำตอบจาก evidence top-5 ด้วย `gemini-3.5-flash-lite`, temperature 0.0 แล้วเทียบกับ `positive_answer` ของคำถามเดียวกัน; ประเมินสำเร็จ 100/100 ต่อ split

| Split | Answer Token F1 | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Citation heuristic correct | Abstained |
|---|---:|---:|---:|---:|---:|---:|
| Validation, n=100 | 0.395647 | 0.395647 | 0.281292 | 0.302869 | 0.640000 (64/100) | 0.280000 (28/100) |
| Test, n=100 | 0.487709 | 0.487709 | 0.390759 | 0.371400 | 0.800000 (80/100) | 0.130000 (13/100) |

ที่มา: [answer_metrics_validation_100.csv](../results/answer_metrics_validation_100.csv), [answer_metrics_test_100.csv](../results/answer_metrics_test_100.csv); แต่ละไฟล์เก็บคะแนนรายคำถาม ส่วนค่าในตารางคือค่าเฉลี่ยที่คำนวณโดย [evaluate_answers.py](../scripts/evaluate_answers.py). Prediction อยู่ใน [qa_predictions_validation_100.jsonl](../results/qa_predictions_validation_100.jsonl) และ [qa_predictions_test_100.jsonl](../results/qa_predictions_test_100.jsonl). Validation prediction file มี 200 records เนื่องจากเก็บทั้ง error และ success; evaluator กรอง error ออก เหลือ success 100 รายการ

ข้อจำกัด: คะแนนข้อความวัดความคล้ายกับ reference ไม่รับรองความถูกต้องของข้อกฎหมาย; citation score เป็น substring heuristic ไม่ใช่ expert review. คำตอบ validation/test เป็นตัวอย่าง 100 ข้อ ไม่ใช่ทั้ง split

## Category Filter

เปรียบเทียบ Hybrid MiniLM + reranker แบบไม่กรองกับแบบกรองเอกสารจากหมวด oracle บน validation 100 คำถาม ใช้ top-20 ก่อน rerank และ top-5 หลัง rerank; มี 1 ข้อไม่มี ground truth จึงคำนวณจริง 99 ข้อ

| Setting | Recall@5 | คำถามที่คิดคะแนน | ข้าม: ไม่มี ground truth | ข้าม: เอกสารไม่อยู่ใน corpus |
|---|---:|---:|---:|---:|
| ไม่กรอง | 0.749158 | 99 | 1 | 0 |
| Oracle category filter | 0.814815 | 99 | 1 | 0 |

ที่มา: [category_filter_recall.csv](../results/category_filter_recall.csv), รันด้วย [evaluate_category_checks.py](../scripts/evaluate_category_checks.py). Oracle category คือหมวด auto-category เสียงข้างมากของ positive contexts ที่ map เข้า corpus; ทั้งสอง setting วัด recall ของ positive contexts ในหมวดนั้น จึงจำลองว่าผู้ใช้เลือกหมวดได้ถูก ไม่ใช่ผลจริงของ category classifier และไม่ใช่ recall ต่อ positive contexts ทุกหมวด

## Runtime และ Memory

วัด 100 validation questions โดยรันแต่ละวิธีใน process แยก; มี warm-up หนึ่งคำถามก่อนจับเวลา ค่า latency เฉลี่ยไม่รวม model loading ส่วน peak RSS รวมการโหลดเอกสาร/โมเดลและ warm-up

| วิธี | Latency เฉลี่ย (ms/คำถาม) | Peak RSS (MB) | คำถาม |
|---|---:|---:|---:|
| BM25 | 24.818 | 1,120.273 | 100 |
| MiniLM | 12.706 | 2,144.418 | 100 |
| Hybrid MiniLM | 30.207 | 2,229.637 | 100 |
| BGE-M3 | 23.026 | 3,619.578 | 100 |
| Hybrid MiniLM + reranker | 136.590 | 2,399.734 | 100 |

ที่มา: [runtime_benchmark.csv](../results/runtime_benchmark.csv), [benchmark_runtime.py](../scripts/benchmark_runtime.py). บันทึกเดิมระบุว่ารันใน Colab environment เดียวกันทั้งห้าวิธี แต่ CSV ไม่เก็บ hardware identifier; ใช้เทียบเฉพาะภายใต้รอบ/environment นี้ ไม่ควรอ้างเป็น benchmark ข้ามเครื่อง

## การตรวจตัวอย่างและผลที่ยังไม่ยืนยัน

- Retrieval spot-check 50 คำถาม validation: รายงาน R@1=0.68, R@3=0.80, R@5=0.82, MRR=0.7407; พบหนึ่งข้อไม่มี positive context และเมื่อตัดกรณีนั้นออก มี 41/49 = 83.7% ที่พบ gold context ใน top-5. เป็น sample inspection ไม่ใช่ผล full validation. ที่มา: [human_eval_50_results.jsonl](../results/human_eval_50_results.jsonl), [project-summary.md](project-summary.md), [error-analysis.md](../results/error-analysis.md)
- Error analysis ในตัวอย่างเดียวกันบันทึก retrieval misses 8 รายการ และ data issue 1 รายการ (positive answer มีแต่ positive context ว่าง); รายละเอียดเป็นกรณีศึกษา ไม่ใช่อัตราความผิดพลาดของทั้งชุด: [error-analysis.md](../results/error-analysis.md)
- Answer human-evaluation rubric มี 40 ตัวอย่าง แบ่ง validation/test อย่างละ 20 แต่คะแนนมนุษย์ยังว่าง; ไฟล์ [human_eval_ai_review_draft.jsonl](../results/human_eval_ai_review_draft.jsonl) เป็น AI draft ไม่ใช่ผล human evaluation ที่ยืนยันแล้ว ดู [human-evaluation-guide.md](human-evaluation-guide.md)
- Category review template มีช่อง human category/correct ว่าง จึงยังไม่มี category accuracy จากผู้ประเมิน; [category_ai_review_draft.csv](../results/category_ai_review_draft.csv) เป็นข้อเสนอจาก rule classifier และ `suggestion_matches_auto` ไม่ใช่ accuracy หรือการตรวจอิสระ ดู [category-review-guide.md](category-review-guide.md)

## ไฟล์ผลเพิ่มเติมและการตีความ

- [retrieval_metrics.csv](../results/retrieval_metrics.csv) และแถวที่ตรงกันใน [retrieval_test_all.csv](../results/retrieval_test_all.csv) เป็น Hybrid test metrics ชุดเดียวกับ final baseline ไม่ใช่การทดลองเพิ่ม
- [reranker_test_all.csv](../results/reranker_test_all.csv) ซ้ำผล test เต็มชุดที่สรุปใน [final_reranker_test_results.csv](../results/final_reranker_test_results.csv); ไฟล์ `reranker_validation_chunks.csv` และ `reranker_test_chunks.csv` เป็น metrics แยกช่วงไม่เกิน 100 คำถาม (ช่วงสุดท้ายอาจสั้นกว่า) ไม่ควรนำแถวเหล่านั้นมาแทนค่า aggregate ของ split
- [hybrid_bge_m3_results.csv](../results/hybrid_bge_m3_results.csv) มี alpha sweep (0.25, 0.5, 0.75) โดย Recall@1 เท่ากับ 0.608498, 0.663816, 0.701497; Recall@3 เท่ากับ 0.753608, 0.800641, 0.839925; Recall@5 เท่ากับ 0.799038, 0.846339, 0.879476; MRR เท่ากับ 0.695531, 0.746172, 0.781402 ตามลำดับ แต่ไฟล์ไม่มี split metadata และ experiment script โหลด test parquet จึงจัดเป็นผลสำรวจที่ยังไม่ยืนยัน ไม่ใช้แทน validation benchmark หรืออ้างเป็น official test result
- [answer_metrics.csv](../results/answer_metrics.csv) เป็นคะแนนรายคำถามที่ไม่มี split metadata ชัดเจน; ใช้ไฟล์แยก validation/test 100 ข้อที่มี prediction/reference ระบุชัดสำหรับตัวเลขสรุปในเอกสารนี้

## วิธีรันซ้ำ

ตัวอย่างคำสั่งที่สร้างผลหลัก:

```bash
python scripts/benchmark_retrievers.py --split validation --methods bm25 minilm hybrid_minilm
python scripts/evaluate_reranker.py --model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
python scripts/evaluate_retrieval_f1.py --limit 100 --methods bm25 minilm hybrid hybrid_reranker --output results/retrieval_f1_test_100.csv
python scripts/evaluate_context.py --questions data/processed/validation_retrieval.parquet --limit 100 --output results/context_metrics.csv
python scripts/evaluate_answers.py --predictions results/qa_predictions_validation_100.jsonl --references data/processed/validation_retrieval.parquet --output results/answer_metrics_validation_100.csv
python scripts/benchmark_runtime.py --limit 100 --methods bm25 minilm hybrid bge_m3 hybrid_reranker
```

การรันซ้ำบางรายการต้องมีไฟล์ data/embeddings, dependencies หรือ Gemini predictions ที่เตรียมไว้ก่อน คำสั่งสร้างคำตอบต้องตั้ง `GEMINI_API_KEY` ใน terminal และไม่ควรบันทึก key ลง repository