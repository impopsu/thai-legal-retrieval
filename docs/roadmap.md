# แผนงานโครงการ

## ขอบเขตปัจจุบัน

โครงการมี retrieval core ที่นำกลับมาใช้ได้ใน `legal_qa/` และเก็บ scripts เดิมไว้ใน
`experiments/` เพื่อการเปรียบเทียบที่ทำซ้ำได้ convention หลักคือ:

```text
hybrid_score = alpha * normalized_bm25 + (1 - alpha) * normalized_semantic
```

ยังไม่มี retrieval method ใดถูกเลือกเป็นระบบสุดท้าย BM25, Hybrid MiniLM, BGE-M3
และ Hybrid BGE-M3 ต้องทำ benchmark ภายใต้ validation/test protocol เดียวกันก่อน
เลือก retriever หลัก โดยรายงาน runtime และ memory ควบคู่กับคุณภาพการ retrieval
เนื่องจาก BGE-M3 มีค่าใช้จ่ายสูงบน CPU

## หมุดหมาย

1. Retrieval benchmark: เปรียบเทียบ BM25, Hybrid MiniLM, BGE-M3 และ Hybrid BGE-M3
2. Reranker benchmark: rerank candidate set เดียวกันและเปรียบเทียบกับ retrieval baselines ที่เลือกไว้บน validation
3. Final method selection: เลือก retrieval และ reranker โดยใช้ validation เท่านั้น
4. Evidence pipeline: ส่งคืน legal documents ที่จัดอันดับแล้วพร้อมชื่อกฎหมายและมาตรา
5. Grounded QA: สร้างคำตอบจาก retrieved evidence เท่านั้นและใส่ citations
6. QA evaluation: วัด answer correctness, citation correctness และ faithfulness
7. Demo: เปิดให้ใช้งาน pipeline ที่เลือกผ่าน CLI หรือ web interface

มีการ implement answer-level evaluation แล้ว แต่ต้องมีคำตอบที่สร้างในรูปแบบ JSONL
โครงการจะไม่สร้าง answer metrics ขึ้นเองก่อนที่จะเลือก LLM หรือ generator อื่น

โครงการพร้อมสำหรับรายงานผล retrieval และ grounded-prompt experiments แล้ว แต่ผล
ด้านคุณภาพคำตอบจริงยังต้องเลือก generator และเก็บผลการตัดสินจากมนุษย์โดยใช้
`docs/human-evaluation.md`

Reranking เป็นการทดลองหลักเพื่อปรับปรุง retrieval ที่อาศัย similarity พื้นฐาน
ต้องประเมินบน validation ก่อนนำไปใช้ใน final test configuration และไม่ได้ถือว่า
เป็นส่วนเสริมที่เลือกใช้ได้ตามอำเภอใจ

## คำสั่งประเมินผลปัจจุบัน

สร้าง validation data จาก training split เท่านั้น:

```bash
python scripts/split_train_validation.py
```

เลือก `alpha` โดยใช้ validation:

```bash
python scripts/select_alpha.py
```

เปรียบเทียบ retrieval methods บน validation:

```bash
python scripts/benchmark_retrievers.py \
	--split validation \
	--methods bm25 minilm hybrid_minilm
```

รันการประเมิน final test ด้วย alpha ที่เลือกไว้ ห้ามใช้คำสั่งนี้เพื่อ tune parameters:

```bash
python scripts/evaluate_retrieval.py --methods bm25 hybrid --alpha 0.5
```

## หมายเหตุเกี่ยวกับการทำซ้ำผลการทดลอง

- รัน scripts จาก root ของ repository
- ใช้ validation เพื่อเลือก `alpha` และใช้ test สำหรับการรายงานผลสุดท้ายเท่านั้น
- ให้ถือว่า `MRR@5` เป็น metric ของ top-five ไม่ใช่ full-ranking MRR
- โดยค่าเริ่มต้น Git จะเพิกเฉยต่อ generated datasets, embeddings และ result files