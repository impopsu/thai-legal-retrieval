# ระบบตอบคำถามและ Retrieval ด้านกฎหมายไทย

## Slide 1 — ชื่อเรื่อง

**Thai Legal Question Answering and Retrieval System**

Thai Legal QA / RAG

Computer Engineering  
Kasetsart University

---

## Slide 2 — ปัญหา

### ปัญหา

คำถามกฎหมายไทยมักต้องค้นหามาตรากฎหมายที่ตรงกันก่อนจึงจะสร้างคำตอบได้

ความท้าทาย:

- มาตรากฎหมายมีศัพท์ที่คล้ายกัน
- อาจระบุกฎหมายได้ถูกต้องแต่พลาดมาตราที่ตรงกัน
- semantic หรือ lexical retrieval เพียงอย่างเดียวอาจคืนมาตราที่เกี่ยวข้องใกล้เคียงแต่ไม่ถูกต้อง
- คำตอบจาก LLM ต้องมี legal evidence ที่เชื่อถือได้เพื่อลด hallucination

### เป้าหมาย

ปรับปรุงการ retrieval legal evidence และจัดเตรียม grounded evidence สำหรับการสร้างคำตอบ

---

## Slide 3 — ชุดข้อมูล

### WangchanX-Legal-ThaiCCL-RAG

ชุดข้อมูล Thai legal QA/RAG ที่มุ่งเน้นกฎหมายบริษัทและกฎหมายพาณิชย์

**ข้อมูลภายในเครื่อง**

- 35 laws
- 4,545 unique legal sections
- 8,211 training questions
- 1,643 validation questions
- 3,742 test questions

แต่ละ QA example ประกอบด้วย:

- Question
- Positive context
- Hard-negative context
- Positive answer
- Hard-negative answer

---

## Slide 4 — สถาปัตยกรรมระบบ

### Pipeline ที่เสนอ

**Question**

↓

**Hybrid Retrieval**

BM25 + Semantic Retrieval

↓

**Top-20 Candidates**

↓

**Cross-Encoder Reranker**

↓

**Top-5 Evidence**

↓

**Grounded LLM**

↓

**Answer + Citation**

ขั้นตอนแรก retrieval candidates อย่างมีประสิทธิภาพ ส่วน reranker ทำ fine-grained relevance ranking

---

## Slide 5 — วิธีการ Retrieval

### Methods ที่เปรียบเทียบ

**BM25**

ใช้ Lexical retrieval โดยอาศัยการจับคู่คำศัพท์

**MiniLM**

ใช้ Dense semantic retrieval ด้วย sentence embeddings

**Hybrid**

รวม BM25 และ semantic retrieval เข้าด้วยกัน

**BGE-M3**

ประเมิน Multilingual embedding model ในฐานะ semantic retrieval baseline ที่มีประสิทธิภาพสูงขึ้น

---

## Slide 6 — Validation: การเลือก Hybrid Weight

### Hybrid Alpha

ใช้ validation set เพื่อเลือก weighting parameter

validation MRR ที่ดีที่สุดได้จาก:

**alpha = 0.5**

ใช้ configuration นี้สำหรับ final Hybrid retrieval pipeline

---

## Slide 7 — ผลการทดสอบ

### ประสิทธิภาพ Retrieval

| Method | Split | R@1 | R@3 | R@5 | MRR |
|---|---|---:|---:|---:|---:|
| BM25 | Test | 55.59% | 70.12% | 75.28% | 0.6328 |
| Semantic MiniLM | Test | 33.54% | 48.61% | 55.13% | 0.4168 |
| Hybrid α=0.5 | Test | 58.82% | 74.35% | 79.90% | 0.6701 |
| Hybrid MiniLM + Cross-Encoder Reranker (FINAL) | Test | 71.94% | 83.54% | 86.26% | 0.7790 |
| BGE-M3 | Validation | 65.43% | 81.13% | 86.73% | 0.7379 |

**ข้อสังเกต**

final pipeline คือ Hybrid MiniLM + Cross-Encoder Reranker ส่วน BGE-M3 แสดงไว้เฉพาะในฐานะ validation comparison และไม่มีการอ้างผล BGE-M3 test-set

---

## Slide 8 — การวิเคราะห์ข้อผิดพลาด

### การตรวจสอบจากตัวอย่าง

ตรวจสอบ validation questions จำนวน 50 ข้อ

ข้อผิดพลาดที่สังเกตพบ:

- มาตราผิดภายในกฎหมายฉบับเดียวกัน
- ศัพท์กฎหมายที่คล้ายกัน
- กฎหมายที่เกี่ยวข้องแต่ไม่ถูกต้อง
- gold section ที่ตรงกันถูกจัดอันดับต่ำกว่ามาตราที่เกี่ยวข้อง
- ตัวอย่างหนึ่งในชุดข้อมูลไม่มี positive context

### ข้อค้นพบหลัก

ข้อผิดพลาดจำนวนมากเป็น **fine-grained ranking errors** ไม่ใช่ retrieval failures โดยสมบูรณ์

ข้อค้นพบนี้สนับสนุนการใช้ Cross-Encoder Reranker ในขั้นตอนที่สอง

---

## Slide 9 — Grounded QA

### การสร้างคำตอบแบบ Grounded

ส่ง legal sections ที่ retrieval ได้ระดับ top-5 ให้ LLM เป็น evidence

prompt กำหนดให้ model:

- ใช้เฉพาะ retrieved evidence
- หลีกเลี่ยงการแต่งเติมบทบัญญัติกฎหมาย
- ระบุเมื่อ evidence ไม่เพียงพอ
- ระบุกฎหมายและมาตราที่เกี่ยวข้อง

การออกแบบนี้แยก **retrieval** ออกจาก **answer generation**

---

## Slide 10 — ข้อจำกัดของการประเมิน

### การประเมินระดับคำตอบ

มีการ implement answer-generation pipeline และ evaluator

Metrics:

- Token-level F1
- Citation correctness
- Abstention

อย่างไรก็ตาม ไม่สามารถประเมิน answer-level แบบเต็มได้เนื่องจาก free Gemini API quota ที่มีอยู่หมดลง

ดังนั้น:

**Retrieval performance เป็นการประเมินเชิงปริมาณหลัก**

---

## Slide 11 — Demo

### การสาธิตระบบ

สาธิต:

1. ป้อนคำถามกฎหมายไทย
2. retrieval มาตรากฎหมายที่เกี่ยวข้อง
3. rerank candidate evidence
4. สร้าง grounded answer
5. แสดง legal source ที่อ้างอิง

---

## Slide 12 — บทสรุป

### บทสรุป

โครงการนี้ implement Thai legal QA/RAG pipeline ที่มี multi-stage retrieval

ผลลัพธ์สำคัญ:

- BM25 เป็น lexical baseline ที่แข็งแรง
- Hybrid retrieval ปรับปรุงผลจาก BM25
- Hybrid MiniLM + Cross-Encoder Reranker คือ FINAL PIPELINE โดยมี test-set R@1 71.94%, R@3 83.54%, R@5 86.26% และ MRR 0.7790
- BGE-M3 เป็น validation comparison ไม่ใช่ final pipeline
- การวิเคราะห์ข้อผิดพลาดแสดงให้เห็นว่าการจัดอันดับมาตราที่ตรงกันยังเป็นความท้าทาย
- Cross-Encoder reranking เป็นแนวทางที่เหมาะสมสำหรับขั้นตอนที่สอง
- Grounded prompting เชื่อม legal evidence ที่ retrieval ได้เข้ากับการสร้างคำตอบของ LLM

### งานในอนาคต

- การประเมินโดยผู้เชี่ยวชาญแบบเต็ม
- การประเมิน citation ที่ละเอียดขึ้น
- การวิเคราะห์ latency และ memory
- การเปรียบเทียบ reranker models เพิ่มเติม
- การสร้างและประเมินคำตอบแบบเต็มขนาด
