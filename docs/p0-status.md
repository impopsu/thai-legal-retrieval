# สถานะ P0

## Validation benchmark

วิธีการทั้งหมดด้านล่างได้รับการประเมินบน validation split เดียวกันจำนวน 1,643
คำถาม โดยใช้ processed document index เดียวกันและนิยาม Recall/MRR เดียวกัน

| Method | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---|---:|---:|---:|---:|
| BM25 | 0.4948 | 0.6519 | 0.7158 | 0.5783 |
| Semantic MiniLM | 0.3214 | 0.4729 | 0.5508 | 0.4061 |
| Hybrid MiniLM (alpha=0.5) | 0.5673 | 0.7206 | 0.7669 | 0.6462 |
| BGE-M3 | 0.6543 | 0.8113 | 0.8673 | 0.7379 |
| Hybrid BGE-M3 (alpha=0.5) | 0.6409 | 0.7815 | 0.8411 | 0.7166 |

ค่า Hybrid BGE-M3 ด้านบนเป็นผล validation ที่ใช้อ้างอิงจาก
`results/retrieval_benchmark.csv` ซึ่งสร้างโดย `scripts/benchmark_retrievers.py`
ด้วย validation split จำนวน 1,643 คำถาม ส่วน
`results/hybrid_bge_m3_results.csv` ถูกสร้างโดย experiment script ที่โหลด test parquet
ซึ่งไม่ได้ถูกแตะต้อง และไม่ได้บันทึก split ไว้ใน output ดังนั้นค่า R@1 ที่ alpha `0.5`
เท่ากับ `0.6638` จึงไม่ใช่ผล validation และไม่นำมาใช้ในตารางนี้

## การตีความ

- ปัจจุบัน BGE-M3 เป็น retrieval candidate ที่มีผลดีที่สุดบน validation
- Hybrid MiniLM ปรับปรุงผลจาก BM25 ได้โดยใช้ model ที่มีขนาดเล็กกว่ามาก
- Hybrid BGE-M3 ที่มี alpha `0.5` ไม่ได้ปรับปรุงผลเมื่อเทียบกับ BGE-M3 เพียงอย่างเดียว
- ไม่ควรเลือก final method จาก validation เพียงอย่างเดียวโดยไม่บันทึก trade-off ด้าน runtime และ memory

## Reranker validation

ประเมิน reranker บน validation questions ทั้ง 1,643 ข้อ โดยใช้ Hybrid
MiniLM, alpha `0.5`, candidate top-20 และ model
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`:

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---|---:|---:|---:|---:|
| Hybrid MiniLM baseline | 0.5673 | 0.7206 | 0.7669 | 0.6462 |
| Hybrid MiniLM + reranker | 0.6999 | 0.8223 | 0.8497 | 0.7618 |

reranker ปรับปรุง retrieval metric ที่รายงานทุกค่าในการประเมินบน validation split
แบบเต็ม จึงถูกเลือกสำหรับ final test configuration

## Final test configuration

```text
Retriever: Hybrid MiniLM
Alpha: 0.5 (selected on validation)
Candidates: top-20
Reranker: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
Final evidence: top-5
```

configuration นี้ได้รับการประเมินหนึ่งครั้งบน official test split ที่ไม่ถูกแตะต้อง
จำนวน 3,742 คำถาม:

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---|---:|---:|---:|---:|
| Hybrid MiniLM baseline | 0.5882 | 0.7435 | 0.7990 | 0.6701 |
| Hybrid MiniLM + reranker | 0.7194 | 0.8354 | 0.8626 | 0.7790 |

ผล BGE-M3 validation ยังคงเป็น comparison candidate โดยไม่ได้ใช้เพื่อ tune final
test configuration และไม่มีการทำ test tuning เพิ่มเติม

## ข้อจำกัดของ Demo

final CLI pipeline เชื่อมต่ออย่างถูกต้องแล้ว แต่ interactive query ยังอาจ retrieval
evidence ที่อ่อน แม้ aggregate benchmark metrics จะแข็งแรง ตัวอย่างเช่น informal
query `ถ้าขโมยของคนอื่น มีความผิดอะไร` ไม่ได้คืน theft provision ที่เกี่ยวข้องอย่างชัดเจน
ในการ smoke test กรณีนี้ถูกบันทึกเป็นปัญหา error-analysis และ corpus-coverage โดยระบบ
ต้องไม่สร้าง legal answer เมื่อ evidence ไม่เกี่ยวข้องอย่างชัดเจน

## คำสั่ง

Validation benchmark:

```bash
python scripts/benchmark_retrievers.py \
  --split validation \
  --methods bm25 minilm hybrid_minilm bge hybrid_bge \
  --alpha 0.5
```

Reranker validation:

```bash
python scripts/evaluate_reranker.py \
  --model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
  --retriever hybrid \
  --alpha 0.5
```