# Project Structure

| โฟลเดอร์ | เอาไว้ทำอะไร |
|---|---|
| `data/raw` | Dataset ต้นฉบับที่ยังไม่แก้ |
| `data/processed` | Dataset ที่เรา preprocess แล้ว |
| `legal_qa` | Core retrieval, reranking, grounded QA และ evaluation ใน package เดียว |
| `scripts` | คำสั่งเตรียมข้อมูล ประเมินผล และเปิด demo |
| `experiments/baseline` | โค้ดและผลการทดลองของระบบเดิมที่ใช้เป็น Baseline |
| `experiments/bge_m3` | การทดลองที่เกี่ยวข้องกับ BGE-M3 |
| `experiments/reranker` | การทดลองที่เกี่ยวข้องกับ Reranker |
| `results` | ผลลัพธ์จากการทดลองและการประเมินระบบ |
| `docs` | Paper, Dataset, Methodology และบันทึกการทำงานของโปรเจกต์ |
| `notebooks` | Jupyter Notebook สำหรับทดลอง วิเคราะห์ข้อมูล และทดสอบแนวคิด |
