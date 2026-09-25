# สรุปโครงงานสำหรับปรึกษาอาจารย์

## 1. ชื่อโครงงาน

**Thai Legal Question Answering and Retrieval System**

เป็นโครงงานระบบตอบคำถามและ Retrieval ด้านกฎหมายไทย โดยมุ่งเน้นการค้นหา legal sections ที่เกี่ยวข้องและจัดเตรียม evidence สำหรับการสร้างคำตอบแบบ grounded

## 2. ที่มาและปัญหา

คำถามกฎหมายไทยมักต้องค้นหามาตราที่ตรงกันก่อนจึงจะสามารถสร้างคำตอบที่น่าเชื่อถือได้ ปัญหาสำคัญคือมาตรากฎหมายจำนวนมากมีศัพท์และเนื้อหาที่คล้ายกัน ระบบอาจระบุกฎหมายได้ถูกฉบับแต่จัดอันดับมาตราที่ผิด หรือคืนกฎหมายที่เกี่ยวข้องใกล้เคียงแต่ไม่ใช่คำตอบที่ตรงกัน

โครงงานจึงศึกษาการปรับปรุง legal document Retrieval ด้วยวิธี lexical retrieval, semantic retrieval, Hybrid Retrieval และ Cross-Encoder Reranking โดยมีเป้าหมายให้ LLM ใช้ evidence ที่เกี่ยวข้องแทนการสร้างคำตอบโดยไม่มีหลักฐานรองรับ

## 3. วัตถุประสงค์

- พัฒนาระบบ Retrieval สำหรับคำถามกฎหมายไทย
- เปรียบเทียบ BM25, Semantic MiniLM, Hybrid MiniLM, BGE-M3 และ Hybrid BGE-M3
- เลือกค่า `alpha` และ configuration จาก Validation โดยไม่ใช้ Test Set ในการ tune
- ประเมิน Cross-Encoder Reranker สำหรับการจัดอันดับ candidate legal sections
- สร้าง grounded QA pipeline ที่ส่ง evidence ให้ LLM
- ประเมิน Retrieval ด้วย Recall@1, Recall@3, Recall@5 และ MRR
- ตรวจสอบข้อผิดพลาดจากตัวอย่าง Retrieval และวางแนวทาง Human Evaluation

## 4. Dataset

### ชุดข้อมูลและแหล่งที่มา

- Dataset: **WangchanX-Legal-ThaiCCL-RAG**
- Source: [Hugging Face: airesearch/WangchanX-Legal-ThaiCCL-RAG](https://huggingface.co/datasets/airesearch/WangchanX-Legal-ThaiCCL-RAG)
- ลักษณะงาน: Thai legal question answering และ Retrieval-Augmented Generation
- เนื้อหาเน้นกฎหมายบริษัทและกฎหมายพาณิชย์

### ข้อมูลที่มีในแต่ละ record

```text
question
positive_contexts
hard_negative_contexts
positive_answer
hard_negative_answer
```

### ขนาดและการแบ่งข้อมูล

ข้อมูลที่รายงานในเอกสารโครงงานมีดังนี้:

- Corpus ภายในเครื่อง: legal sections ที่ไม่ซ้ำกัน 4,545 มาตรา จากกฎหมาย 35 ฉบับ
- Training: 8,211 questions
- Validation: 1,643 questions
- Test: 3,742 questions

ไฟล์หลักภายในเครื่องคือ `data/raw/train-00000-of-00001.parquet` และ `data/raw/test-00000-of-00001.parquet` ส่วนดัชนีเอกสารภายในเครื่องสร้างไว้ใน `data/processed/legal_documents.csv` จาก context records

ตาม protocol ของโครงงาน official test split ถูกเก็บไว้โดยไม่แตะต้อง และใช้สำหรับการรายงานผลสุดท้ายเท่านั้น ส่วน model, `alpha`, top-k และ Reranking settings ต้องเลือกจาก Validation

## 5. ภาพรวมของระบบ

ระบบรับคำถามกฎหมายไทย แล้วใช้ Retrieval เพื่อสร้าง candidate legal sections จากนั้นใช้ Cross-Encoder Reranker จัดอันดับ candidate ก่อนส่ง evidence ที่ดีที่สุดให้ grounded prompt และ LLM

```text
คำถามกฎหมายไทย
        |
        v
Hybrid Retrieval: BM25 + Semantic MiniLM
        |
        v
Top-20 Candidate Legal Sections
        |
        v
Cross-Encoder Reranker
        |
        v
Top-5 Evidence
        |
        v
Grounded LLM Answer + Citation
```

ใน final configuration ใช้ Hybrid MiniLM ที่ `alpha = 0.5`, candidate top-20 และ Reranker model `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` จากนั้นเลือก evidence top-5

## 6. โครงสร้างโปรเจกต์

| ส่วนประกอบ | หน้าที่ |
|---|---|
| `data/raw` | Dataset ต้นฉบับ รวม train/test Parquet |
| `data/processed` | Dataset และ document index ที่ผ่านการประมวลผล |
| `legal_qa/retrieval.py` | BM25, Semantic Retrieval และ Hybrid Retrieval |
| `legal_qa/reranking.py` | Cross-Encoder Reranker |
| `legal_qa/evaluation.py` | การคำนวณ Retrieval metrics และการประเมิน batch |
| `legal_qa/qa.py` | การสร้าง grounded prompt และ answer generation interface |
| `legal_qa/answer_evaluation.py` | การประเมิน answer correctness และ citation |
| `scripts/benchmark_retrievers.py` | Benchmark วิธี Retrieval บน split ที่กำหนด |
| `scripts/evaluate_retrieval.py` | ประเมิน Retrieval configuration |
| `scripts/evaluate_reranker.py` | ประเมิน Reranker บน Validation |
| `scripts/select_alpha.py` | เลือก `alpha` จาก Validation |
| `scripts/split_train_validation.py` | แบ่ง training data เป็น train และ Validation |
| `experiments/baseline` | BM25, Semantic, Hybrid และ baseline experiments |
| `experiments/bge_m3` | BGE-M3 และ Hybrid BGE-M3 experiments |
| `results` | Saved metrics, evaluation outputs และ analysis artifacts |
| `docs` | รายงาน protocol, roadmap, evaluation และ documentation |

## 7. วิธีการที่ทดลอง

### BM25

Lexical Retrieval ที่อาศัย term matching ใช้เป็น baseline สำหรับคำศัพท์กฎหมายที่ตรงกันระหว่าง question และ legal section

### Semantic MiniLM

ใช้ sentence embedding model ที่อาศัย MiniLM แปลง questions และ legal sections เป็น dense vectors แล้วใช้ semantic similarity ในการ Retrieval

### Hybrid MiniLM

รวมคะแนน BM25 และ Semantic Retrieval ด้วย weighted hybrid score ตาม convention:

```text
hybrid_score = alpha * normalized_bm25 + (1 - alpha) * normalized_semantic
```

ค่า `alpha = 0.5` ถูกเลือกจาก Validation และใช้เป็น Retrieval stage ของ final pipeline

### BGE-M3

ใช้ `BAAI/bge-m3` เป็น multilingual embedding model สำหรับการเปรียบเทียบ Semantic Retrieval ที่มีขนาดและความสามารถแตกต่างจาก MiniLM ผลที่อ้างอิงในเอกสารปัจจุบันเป็น Validation comparison ไม่ใช่ final pipeline

### Hybrid BGE-M3

รวม BM25 กับ BGE-M3 embeddings และทดลองค่า `alpha` หลายค่าใน experiment script มีผลลัพธ์จากสองเส้นทางที่ต้องแยกกัน:

- `results/retrieval_benchmark.csv` ระบุชัดว่าเป็น `validation` และเป็นผล authoritative สำหรับ Hybrid BGE-M3 validation
- `results/hybrid_bge_m3_results.csv` สร้างจาก script ที่โหลด `data/raw/test-00000-of-00001.parquet` แต่ output ไม่บันทึก split จึงไม่ควรนำไปเรียกว่า Validation result

### Cross-Encoder Reranker

ใช้ `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` ให้คะแนน question กับ candidate section ร่วมกัน โดยรับ candidate top-20 จาก Hybrid MiniLM และเลือก evidence top-5

## 8. การทดลองและการประเมินผล

### Metrics

- **Recall@1**: มี gold context อยู่ในผลลัพธ์อันดับ 1 หรือไม่
- **Recall@3**: มี gold context อยู่ในผลลัพธ์ 3 อันดับแรกหรือไม่
- **Recall@5**: มี gold context อยู่ในผลลัพธ์ 5 อันดับแรกหรือไม่
- **MRR / MRR@5**: ประเมินอันดับของ gold context โดย MRR@5 จำกัดการพิจารณาไว้ที่ top-five ในรายงานที่ระบุเช่นนั้น

### Validation

ใช้ Validation split จำนวน 1,643 questions เพื่อเลือก `alpha`, model/configuration และ Reranker settings โดยไม่ใช้ official Test Set สำหรับการ tune

Validation ที่ตรวจสอบได้โดยตรงจากผลลัพธ์ ได้แก่:

- `results/validation_alpha_results.csv`: Hybrid MiniLM alpha comparison
- `results/reranker_validation_results.csv`: Hybrid MiniLM baseline เทียบกับ Reranker
- `results/retrieval_benchmark.csv`: BGE-M3 และ Hybrid BGE-M3 โดยระบุ `split=validation`

`docs/p0-status.md` ยังรายงานค่า Validation ของ BM25 และ Semantic MiniLM แต่ไม่พบ CSV เฉพาะแถวเหล่านี้จาก result files ที่มีอยู่ใน repository จึงควรถือเป็นค่าที่รายงานใน status document จนกว่าจะตรวจสอบผลต้นทางเพิ่มเติม

### Test Set

official Test Set มี 3,742 questions และถูกใช้สำหรับ final reporting ผล baseline อยู่ใน `results/final_test_results.csv` ส่วนผล final Reranker อยู่ใน `results/final_reranker_test_results.csv`

### Human Evaluation และ Sampled Verification

มีการตรวจสอบตัวอย่าง 50 questions จาก Validation โดยดู retrieved evidence และ gold positive contexts ผลที่รายงานใน `docs/final-report.md` และคำนวณจาก `results/human_eval_50_results.jsonl` คือ:

- R@1 = 68.0%
- R@3 = 80.0%
- R@5 = 82.0%
- MRR = 0.7407
- หลังตัดกรณีที่มี `positive_contexts` ว่างออก เหลือ 41/49 = 83.7% ที่มี gold context อยู่ใน top five

โครงงานยังมี Human Evaluation Protocol สำหรับให้ผู้ประเมินให้คะแนน answer correctness, evidence faithfulness, citation correctness, completeness และ abstention behavior แต่ยังไม่มีผล expert evaluation แบบเต็มชุดที่ยืนยันไว้ในรายงานนี้

### Error Analysis

ข้อผิดพลาดหลักที่บันทึกไว้ ได้แก่:

- ได้กฎหมายถูกฉบับแต่มาตราผิด
- ได้มาตราที่มีศัพท์กฎหมายคล้ายกัน
- ได้กฎหมายที่เกี่ยวข้องใกล้เคียงแต่ไม่ใช่ target law
- gold section ที่ตรงกันถูกจัดอันดับต่ำกว่ามาตราที่เกี่ยวข้อง
- มีตัวอย่างหนึ่งที่ positive answer มีอยู่ แต่ `positive_contexts` ว่าง

## 9. ตารางผลการทดลอง

### Test Set: Retrieval Baselines

ค่าด้านล่างมาจาก `results/final_test_results.csv` และแสดงเป็นเปอร์เซ็นต์สำหรับ Recall:

| Method | R@1 | R@3 | R@5 | MRR |
|---|---:|---:|---:|---:|
| BM25 | 55.59% | 70.12% | 75.28% | 0.6328 |
| Semantic MiniLM | 33.54% | 48.61% | 55.13% | 0.4168 |
| Hybrid alpha=0.5 | 58.82% | 74.35% | 79.90% | 0.6701 |

### Test Set: Final Reranker Pipeline

ค่าด้านล่างมาจาก `results/final_reranker_test_results.csv` ซึ่งมี `num_questions=3742`, `candidate_k=20`, `top_k=5` และ `alpha=0.5`:

| Configuration | R@1 | R@3 | R@5 | MRR |
|---|---:|---:|---:|---:|
| Hybrid MiniLM baseline | 58.82% | 74.35% | 79.90% | 0.6701 |
| Hybrid MiniLM + Cross-Encoder Reranker | 71.94% | 83.54% | 86.26% | 0.7790 |

### Validation: Hybrid MiniLM และ Reranker

ค่าด้านล่างตรงกับ `results/validation_alpha_results.csv` และ `results/reranker_validation_results.csv`:

| Configuration | R@1 | R@3 | R@5 | MRR@5 |
|---|---:|---:|---:|---:|
| Hybrid MiniLM alpha=0.5 | 0.5673 | 0.7206 | 0.7669 | 0.6462 |
| Hybrid MiniLM + Cross-Encoder Reranker | 0.6999 | 0.8223 | 0.8497 | 0.7618 |

### Validation: BGE-M3 Comparison

ค่าด้านล่างมาจาก `results/retrieval_benchmark.csv` ซึ่งระบุ `split=validation` และ `num_questions=1643`:

| Method | R@1 | R@3 | R@5 | MRR@5 |
|---|---:|---:|---:|---:|
| BGE-M3 | 0.6543 | 0.8113 | 0.8673 | 0.7379 |
| Hybrid BGE-M3 alpha=0.5 | 0.6409 | 0.7815 | 0.8411 | 0.7166 |

ผล Hybrid BGE-M3 ที่ `R@1 = 0.6409` เป็นค่า Validation ที่ใช้ใน `docs/p0-status.md` ส่วนค่า `R@1 = 0.6638` จาก `results/hybrid_bge_m3_results.csv` เป็นผลจากอีก experiment path ที่โหลด Test Parquet แต่ไม่มี split metadata จึงไม่ใช้แทนค่า Validation และไม่ควรสรุปเป็นผล Test Set ที่เปรียบเทียบอย่างเป็นทางการ

### Validation values ที่ยังต้องตรวจสอบต้นทางเพิ่มเติม

`docs/p0-status.md` รายงานค่าต่อไปนี้:

| Method | R@1 | R@3 | R@5 | MRR@5 |
|---|---:|---:|---:|---:|
| BM25 | 0.4948 | 0.6519 | 0.7158 | 0.5783 |
| Semantic MiniLM | 0.3214 | 0.4729 | 0.5508 | 0.4061 |

ไม่พบแถวผลลัพธ์โดยตรงของสอง method นี้ใน CSV ที่มีอยู่ จึงไม่ยืนยันว่าเป็น saved CSV metrics ใน summary นี้ และไม่ควรใช้เป็นหลักฐานใหม่โดยไม่ตรวจสอบต้นทางหรือ rerun

## 10. Final Pipeline

Final Pipeline ที่เลือกตาม repository evidence คือ:

```text
Hybrid MiniLM (alpha = 0.5)
        -> candidate top-20
Cross-Encoder Reranker
        -> final evidence top-5
Grounded QA / LLM Answer
```

เหตุผลที่เลือก pipeline นี้:

1. `alpha = 0.5` ถูกเลือกจาก Validation
2. Reranker validation ปรับปรุงทุก metric ที่รายงานจาก Hybrid MiniLM baseline
3. ผล Test Set ที่บันทึกไว้ยืนยันว่า Reranker ได้ R@1 = 71.94%, R@3 = 83.54%, R@5 = 86.26% และ MRR = 0.7790
4. `README.md`, `docs/p0-status.md` และโค้ด demo ระบุ Hybrid MiniLM + Cross-Encoder Reranking เป็น pipeline ที่เลือกใช้

BGE-M3 เป็น Validation comparison/alternative experiment เท่านั้น ไม่ใช่ Final Pipeline และไม่มี verified BGE-M3 Test Set result ที่ควรนำมาอ้างเป็นผลสุดท้าย

## 11. ผลจาก Human Evaluation

ผลจาก sampled verification ที่มีอยู่จริง:

- Sample size: 50 validation questions
- R@1 = 68.0%
- R@3 = 80.0%
- R@5 = 82.0%
- MRR = 0.7407
- Valid questions หลังตัดกรณีข้อมูลผิดปกติ: 49
- Gold context อยู่ใน top five: 41/49 = 83.7%

ข้อค้นพบคือความผิดพลาดจำนวนมากเป็น fine-grained ranking errors ระหว่างมาตรากฎหมายที่มีความเกี่ยวข้องกัน ไม่ใช่ความล้มเหลวทั้งหมดในการระบุหมวดกฎหมาย

ยังไม่มีผล Human Evaluation แบบเต็มโดยผู้เชี่ยวชาญที่รายงานเป็นชุด metric สุดท้าย และ answer-level evaluation แบบเต็มไม่เสร็จเนื่องจาก free Gemini API quota หมดระหว่างการทดลอง

## 12. Error Analysis

Error patterns ที่ยืนยันจากเอกสารโครงงานและผล sampled verification ได้แก่:

- Correct law แต่ wrong section
- Similar legal terminology ทำให้เลือกมาตราที่คล้ายกัน
- Related law ถูกเลือกแทน target law
- Exact gold section อยู่ต่ำกว่ามาตราที่เกี่ยวข้อง
- Positive answer มีอยู่แต่ `positive_contexts` ว่างในหนึ่ง record
- Interactive smoke test สำหรับคำถาม `ถ้าขโมยของคนอื่น มีความผิดอะไร` ไม่ได้คืน theft provision ที่เกี่ยวข้องอย่างชัดเจน

ประเด็นเหล่านี้สนับสนุนการใช้ Cross-Encoder Reranker เพื่อปรับปรุง fine-grained relevance ranking และการเพิ่ม corpus coverage ในอนาคต

## 13. สิ่งที่ทำเสร็จแล้ว

- [x] เตรียม Dataset และ document index ภายในเครื่อง
- [x] แบ่งข้อมูลเป็น Training, Validation และ Test ตาม protocol
- [x] Implement BM25 Retrieval
- [x] Implement Semantic MiniLM Retrieval
- [x] Implement Hybrid Retrieval
- [x] ทดลอง BGE-M3 และ Hybrid BGE-M3
- [x] ทดลองเลือก `alpha` จาก Validation
- [x] Implement Cross-Encoder Reranker
- [x] ประเมิน Reranker บน Validation
- [x] ประเมิน final configuration บน untouched Test Set
- [x] บันทึก Retrieval metrics ใน CSV result files
- [x] ทำ sampled retrieval verification
- [x] จัดทำ Error Analysis
- [x] จัดทำ grounded QA prompt และ CLI/web demo
- [x] จัดทำ answer evaluator และ Human Evaluation Protocol
- [x] จัดทำ reproducibility documentation และ project roadmap

## 14. ปัญหาและข้อจำกัด

- Answer-generation pipeline ยังไม่ได้ประเมินแบบเต็ม evaluation set เพราะ API quota จำกัด
- Human Evaluation ที่มีเป็น sampled verification ไม่ใช่ expert evaluation แบบเต็ม
- ผล BGE-M3 Test Set ที่เชื่อถือได้ยังไม่มีการบันทึกไว้ในรูปแบบที่ยืนยันได้
- `results/hybrid_bge_m3_results.csv` ไม่มี split metadata จึงมีความเสี่ยงในการตีความเมื่อเปรียบเทียบกับ `results/retrieval_benchmark.csv`
- ค่า BM25 และ Semantic MiniLM ใน Validation ที่อยู่ใน `docs/p0-status.md` ยังไม่พบ CSV ต้นทางโดยตรงใน result files ที่มีอยู่
- ระบบอาจคืน evidence ที่อ่อนใน interactive query แม้ aggregate benchmark metrics ดี
- BGE-M3 มีค่าใช้จ่ายด้าน CPU สูงกว่า model ขนาดเล็ก
- Dataset มีอย่างน้อยหนึ่งกรณีที่ positive answer มีอยู่แต่ `positive_contexts` ว่าง
- Citation correctness, answer faithfulness, abstention behavior, latency และ memory usage ยังต้องประเมินเพิ่มเติม

## 15. สิ่งที่ยังต้องทำ

งานที่เหลือสอดคล้องกับ roadmap และ status documents ได้แก่:

- ตรวจสอบหรือ rerun ผล Validation ของ BM25 และ Semantic MiniLM ให้มี saved CSV ต้นทางที่ชัดเจน
- หากต้องการเปรียบเทียบ Hybrid BGE-M3 บน Test Set อย่างเป็นทางการ ให้ rerun โดยบันทึก split, configuration และ metadata ให้ชัดเจน
- เก็บ runtime และ memory ของแต่ละ configuration อย่างเป็นระบบ
- ทำ full expert Human Evaluation
- ประเมิน citation correctness, answer faithfulness และ abstention behavior เพิ่มเติม
- เลือก LLM generator และสร้าง answer outputs สำหรับ answer-level evaluation
- ประเมิน latency และ memory usage
- ปรับปรุง corpus coverage และตรวจสอบ failure cases จาก interactive queries
- เปรียบเทียบ Reranker models เพิ่มเติม
- พัฒนาและตรวจสอบ CLI/web demo ต่อไป

## 16. ประเด็นที่ต้องการปรึกษาอาจารย์

1. ควร rerun Validation benchmark ของ BM25 และ Semantic MiniLM เพื่อสร้าง CSV ต้นทางที่ตรวจสอบย้อนกลับได้หรือไม่
2. ควรทำ Hybrid BGE-M3 Test Set experiment ใหม่พร้อม split metadata หรือควรเก็บ BGE-M3 ไว้เป็น Validation comparison เท่านั้น
3. ควรใช้ Recall@5 และ MRR@5 เป็นเกณฑ์หลักของ evidence top-5 หรือควรเพิ่ม metric ด้าน citation/answer quality
4. ควรออกแบบ full expert Human Evaluation อย่างไร และควรใช้ผู้ประเมินกี่คน
5. ควรเลือก LLM generator ใดสำหรับ answer-level evaluation ภายใต้ข้อจำกัดด้าน API quota และทรัพยากร
6. ควรให้ความสำคัญกับการปรับปรุง corpus coverage, Reranker model หรือ latency/memory ก่อน
7. ควรนำผล sampled verification 41/49 = 83.7% ไปใช้เป็น supporting evidence ในรายงานระดับใด
8. ควรปรับ wording และ scope ของการสรุป BGE-M3 อย่างไรเมื่อไม่มี verified Test Set result

## 17. สรุปสำหรับพูดกับอาจารย์

โครงงานนี้เป็นระบบตอบคำถามกฎหมายไทยที่มุ่งปรับปรุงการค้นหามาตรากฎหมายและจัดเตรียม evidence ให้ LLM ระบบทดลอง BM25, Semantic MiniLM, Hybrid MiniLM, BGE-M3, Hybrid BGE-M3 และ Cross-Encoder Reranker โดยใช้ Validation สำหรับเลือก configuration และเก็บ Test Set ไว้รายงานผลสุดท้าย

ผลปัจจุบันเลือก Hybrid MiniLM ที่ `alpha = 0.5` สร้าง candidate top-20 แล้วใช้ Cross-Encoder Reranker เลือก evidence top-5 ผล Test Set ที่ยืนยันจาก `results/final_reranker_test_results.csv` คือ R@1 = 71.94%, R@3 = 83.54%, R@5 = 86.26% และ MRR = 0.7790 ขณะที่ BGE-M3 เป็นเพียง Validation comparison ไม่ใช่ Final Pipeline

นอกจากนี้มี sampled verification 50 ข้อ ได้ R@1 = 68.0%, R@3 = 80.0%, R@5 = 82.0%, MRR = 0.7407 และหลังตัดกรณีข้อมูลผิดปกติพบ gold context ใน top five จำนวน 41/49 หรือ 83.7% ประเด็นที่ยังต้องปรึกษาคือการตรวจสอบ Validation CSV ที่ขาดหาย การทำ Human Evaluation แบบเต็ม การเลือก LLM generator และการปรับปรุง corpus coverage, latency และ answer quality

## แหล่งข้อมูลที่ใช้จัดทำสรุป

### เอกสารโครงการ

- `docs/dataset.md`
- `docs/final-report.md`
- `docs/human-evaluation.md`
- `docs/p0-status.md`
- `docs/presentation.md`
- `docs/project-structure.md`
- `docs/report-outline.md`
- `docs/reproducibility.md`
- `docs/roadmap.md`
- `README.md`

### Source code และ scripts

- `legal_qa/retrieval.py`
- `legal_qa/reranking.py`
- `legal_qa/qa.py`
- `legal_qa/evaluation.py`
- `legal_qa/answer_evaluation.py`
- `scripts/benchmark_retrievers.py`
- `scripts/evaluate_retrieval.py`
- `scripts/evaluate_reranker.py`
- `scripts/select_alpha.py`
- `scripts/split_train_validation.py`

### Result files

- `results/final_test_results.csv`
- `results/final_reranker_test_results.csv`
- `results/validation_alpha_results.csv`
- `results/reranker_validation_results.csv`
- `results/retrieval_benchmark.csv`
- `results/retrieval_metrics.csv`
- `results/human_eval_50_results.jsonl`
- `results/hybrid_bge_m3_results.csv`
