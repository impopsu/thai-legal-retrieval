# ระบบตอบคำถามและ Retrieval ด้านกฎหมายไทย

## 1. บทนำ

โครงการนี้พัฒนาระบบตอบคำถามกฎหมายไทย โดยมุ่งเน้นการปรับปรุงการ retrieval เอกสารกฎหมาย ระบบจะค้นหามาตรากฎหมายที่เกี่ยวข้องกับคำถามกฎหมายไทยแต่ละข้อ และจัดเตรียม evidence ที่ grounded language model สามารถใช้สร้างคำตอบได้

ประเด็นวิจัยหลักคือประสิทธิภาพของ retrieval โดยเฉพาะโครงการนี้เปรียบเทียบ lexical retrieval, semantic retrieval, hybrid retrieval และ pipeline ที่ใช้ reranking

## 2. ชุดข้อมูล

โครงการนี้ใช้ชุดข้อมูล WangchanX-Legal-ThaiCCL-RAG ซึ่งเป็นชุดข้อมูลสำหรับการตอบคำถามและ retrieval ด้านกฎหมายไทย โดยครอบคลุมกฎหมายบริษัทและกฎหมายพาณิชย์

ชุดข้อมูลประกอบด้วย questions, positive contexts, hard-negative contexts, positive answers และ hard-negative answers

corpus ภายในเครื่องมีมาตรากฎหมายที่ไม่ซ้ำกัน 4,545 มาตรา จากกฎหมาย 35 ฉบับ

ชุดข้อมูลภายในเครื่องแบ่งเป็น:

- Training: 8,211 questions
- Validation: 1,643 questions
- Test: 3,742 questions

official test set ไม่ถูกแตะต้องระหว่างการเลือก model และ configuration

## 3. วิธีการ Retrieval

มีการประเมิน retrieval approach หลักสามรูปแบบ

### 3.1 BM25

ใช้ BM25 เป็น lexical retrieval baseline ซึ่งมีประสิทธิภาพเมื่อคำศัพท์กฎหมายสำคัญในคำถามซ้อนทับกับคำศัพท์ในมาตรากฎหมายที่เกี่ยวข้อง

### 3.2 Semantic Retrieval

ใช้ sentence embedding model ที่อาศัย MiniLM แทนคำถามและมาตรากฎหมายเป็น dense vectors จากนั้นใช้ semantic similarity เพื่อ retrieval มาตรากฎหมายที่เกี่ยวข้อง

### 3.3 Hybrid Retrieval

นำ BM25 และ semantic retrieval มารวมกันด้วย weighted hybrid score

ใช้ validation set เพื่อเลือก weighting parameter โดย configuration ที่เลือกคือ alpha = 0.5

## 4. Reranking Pipeline

final QA retrieval pipeline ใช้สองขั้นตอน:

1. Hybrid retrieval สร้าง candidate legal sections จำนวน top-20
2. Cross-Encoder Reranker ให้คะแนนคำถามและ candidate section ร่วมกัน
3. นำ sections ที่ผ่านการ rerank แล้วจำนวน top-5 ไปใช้เป็น evidence ในขั้นตอนการสร้างคำตอบ

pipeline ทั้งหมดคือ:

Question → Hybrid Retrieval → Top-20 Candidates → Cross-Encoder Reranker → Top-5 Evidence → Grounded LLM Answer

การออกแบบสองขั้นตอนนี้ช่วยให้ขั้นตอนแรก retrieval candidate set ที่ครอบคลุมได้อย่างมีประสิทธิภาพ ขณะที่ reranker มุ่งระบุมาตรากฎหมายที่เกี่ยวข้องที่สุด

## 5. การประเมิน Retrieval

ประเมินประสิทธิภาพ Retrieval ด้วย Recall@1, Recall@3, Recall@5 และ Mean Reciprocal Rank (MRR)

| Method | Split | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---:|---:|---:|---:|
| BM25 | Test | 55.59% | 70.12% | 75.28% | 0.6328 |
| Semantic MiniLM | Test | 33.54% | 48.61% | 55.13% | 0.4168 |
| Hybrid alpha=0.5 | Test | 58.82% | 74.35% | 79.90% | 0.6701 |
| Hybrid MiniLM + Cross-Encoder Reranker (FINAL PIPELINE) | Test | 71.94% | 83.54% | 86.26% | 0.7790 |
| BGE-M3 | Validation | 65.43% | 81.13% | 86.73% | 0.7379 |

แถว BGE-M3 เป็นการเปรียบเทียบบน validation-set เท่านั้น รายงานนี้ไม่มีการรายงานผล BGE-M3 test-set ที่ผ่านการตรวจสอบแล้ว

final pipeline คือ Hybrid MiniLM ที่มี alpha = 0.5 ตามด้วย Cross-Encoder Reranker โดยผล test-set แสดงไว้ในตารางด้านบน

## 6. การตรวจสอบ Retrieval จากตัวอย่าง

ตรวจสอบตัวอย่างคำถาม 50 ข้อจาก validation set ด้วยตนเอง โดยพิจารณา retrieved evidence และ gold positive contexts

evidence ชุด top-5 สุดท้ายได้ผลดังนี้:

- Recall@1: 68.0%
- Recall@3: 80.0%
- Recall@5: 82.0%
- MRR: 0.7407

คำถามหนึ่งข้อมี field `positive_contexts` ว่าง แม้จะมี positive answer กรณีนี้ถือเป็นปัญหาของชุดข้อมูล ไม่ใช่ retrieval error ทั่วไป

หลังตัดกรณีนี้ออก มีคำถามที่ valid จำนวน 41 จาก 49 ข้อที่มี gold context อยู่ในผลลัพธ์ top five คิดเป็น 83.7%

ไม่ควรตีความการตรวจสอบจากตัวอย่างนี้ว่าเป็นสิ่งทดแทนการประเมิน benchmark แบบเต็ม

## 7. การวิเคราะห์ข้อผิดพลาด

ข้อผิดพลาดจากตัวอย่างส่วนใหญ่เกิดจากการจัดลำดับมาตรากฎหมายที่เกี่ยวข้องกันไม่ถูกต้อง

รูปแบบข้อผิดพลาดที่พบบ่อย ได้แก่:

- retrieval กฎหมายถูกฉบับแต่ผิดมาตรา
- retrieval มาตราที่มีศัพท์กฎหมายคล้ายกัน
- retrieval กฎหมายที่เกี่ยวข้องใกล้เคียงแทนกฎหมายเป้าหมาย
- ไม่พบ gold section ที่ตรงกัน แม้จะ retrieval มาตราที่เกี่ยวข้องมาได้

ข้อสังเกตเหล่านี้ชี้ว่า retrieval errors ด้านกฎหมายมักเป็น fine-grained ranking errors มากกว่าจะเป็นความล้มเหลวโดยสิ้นเชิงในการระบุหมวดกฎหมายที่เกี่ยวข้อง

ข้อค้นพบนี้เป็นเหตุผลสนับสนุนการใช้ Cross-Encoder Reranker เป็น ranking model ในขั้นตอนที่สอง

## 8. Grounded Question Answering

นำ legal sections ที่ retrieval ได้ในระดับ top-5 ใส่ลงใน grounded prompt

answer generation prompt กำหนดให้ language model:

- ตอบโดยใช้เฉพาะ retrieved legal evidence
- หลีกเลี่ยงการแต่งเติมบทบัญญัติหรือรายละเอียดทางกฎหมาย
- ระบุอย่างชัดเจนเมื่อ retrieved evidence ไม่เพียงพอ
- ระบุกฎหมายและมาตราที่เกี่ยวข้อง

ดังนั้นระบบจึงแยก retrieval ออกจาก answer generation และจัดเตรียม evidence ที่ชัดเจนให้ generation model

## 9. การประเมิน Answer Generation

มีการ implement answer-generation pipeline ที่ใช้ Gemini API พร้อม answer evaluator

evaluator รองรับ:

- Token-level F1 เทียบกับ reference answer
- Citation correctness
- Abstention detection

อย่างไรก็ตาม ไม่สามารถประเมิน answer-level แบบเต็มได้ เนื่องจาก free API quota ที่มีอยู่หมดลงระหว่างการทดลอง ดังนั้นจึงไม่นำคำตอบที่สร้างได้เพียงบางส่วนมาใช้เป็น quantitative measure ของคุณภาพคำตอบสุดท้าย

ด้วยเหตุนี้ retrieval evaluation จึงเป็นการประเมินเชิงปริมาณหลักของโครงการนี้

## 10. ข้อจำกัด

ยังมีข้อจำกัดหลายประการ

ประการแรก ไม่ได้ประเมินขั้นตอน answer-generation บน evaluation set ทั้งหมดเนื่องจากข้อจำกัดด้าน API quota

ประการที่สอง human evaluation จำกัดอยู่ที่การตรวจสอบ retrieval จากตัวอย่าง ไม่ใช่การประเมินโดยผู้เชี่ยวชาญแบบเต็มขนาด

ประการที่สาม การทดลองปัจจุบันมุ่งเน้นคุณภาพ retrieval เป็นหลัก การประเมิน citation correctness, answer faithfulness, abstention behavior, latency และ memory usage ที่ละเอียดขึ้นจะช่วยให้การประเมินระบบแข็งแรงยิ่งขึ้น

สุดท้าย ชุดข้อมูลเองมีอย่างน้อยหนึ่งตัวอย่างที่มี positive answer แต่ field ของ positive context ที่สอดคล้องกันว่างอยู่ แสดงให้เห็นถึงความจำเป็นในการคำนึงถึงคุณภาพชุดข้อมูลระหว่างการประเมิน

## 11. บทสรุป

โครงการนี้ implement และประเมิน pipeline สำหรับ retrieval และการตอบคำถามกฎหมายไทย

การทดลองเปรียบเทียบ BM25, semantic retrieval, hybrid retrieval, BGE-M3 และ Cross-Encoder reranking โดยประเมิน BGE-M3 ในฐานะ validation comparison และได้ Recall@1 เท่ากับ 65.43%, Recall@3 เท่ากับ 81.13%, Recall@5 เท่ากับ 86.73% และ MRR เท่ากับ 0.7379 ไม่มีการอ้างผล BGE-M3 test-set

FINAL PIPELINE คือ Hybrid MiniLM ที่มี alpha = 0.5 และ Cross-Encoder Reranker บน untouched test set ได้ Recall@1 เท่ากับ 71.94%, Recall@3 เท่ากับ 83.54%, Recall@5 เท่ากับ 86.26% และ MRR เท่ากับ 0.7790 โดย pipeline นี้สร้าง candidates และเลือก final evidence สำหรับ grounded answer generation

การวิเคราะห์ข้อผิดพลาดชี้ว่า retrieval failures จำนวนมากเกิดขึ้นระหว่างมาตราที่มีความคล้ายคลึงกันทางกฎหมาย ซึ่งสนับสนุนการใช้ reranking เพื่อปรับปรุง fine-grained relevance ranking

โครงการยังมี CLI demonstration, web demonstration, grounded prompting pipeline, answer evaluation tools, reproducibility documentation และ retrieval error analysis
