# โครงสร้างโครงการ

| โฟลเดอร์ | หน้าที่ |
|---|---|
| `data/raw` | Dataset ต้นฉบับที่ยังไม่แก้ |
| `data/processed` | Dataset ที่เรา preprocess แล้ว |
| `legal_qa` | แกนหลักของ retrieval, reranking, grounded QA และ evaluation ใน package เดียว |
| `scripts` | คำสั่งเตรียมข้อมูล ประเมินผล และเปิด demo |
| `experiments/baseline` | โค้ดและผลการทดลองของระบบเดิมที่ใช้เป็น Baseline |
| `experiments/bge_m3` | การทดลองที่เกี่ยวข้องกับ BGE-M3 |
| `legal_qa/reranking.py` | implementation ของ Cross-Encoder Reranker |
| `scripts/evaluate_reranker.py` | คำสั่ง validation และ evaluation ของ Reranker |
| `results` | ผลลัพธ์จากการทดลองและการประเมินระบบ |
| `docs` | Paper, Dataset, Methodology และบันทึกการทำงานของโปรเจกต์ |
