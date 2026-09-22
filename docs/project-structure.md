# Project Structure

| โฟลเดอร์ | เอาไว้ทำอะไร |
|---|---|
| `data/raw` | Dataset ต้นฉบับที่ยังไม่แก้ |
| `data/processed` | Dataset ที่เรา preprocess แล้ว |
| `src/retrieval` | โค้ดสำหรับ BM25, BGE-M3, Hybrid Search และ Reranker |
| `src/evaluation` | โค้ดสำหรับคำนวณ R@1, R@3, R@5, MRR และ metrics อื่น ๆ |
| `src/utils` | ฟังก์ชันช่วยที่ใช้ร่วมกันในโปรเจกต์ |
| `scripts` | ไฟล์ที่ใช้สั่งรัน pipeline หรือกระบวนการต่าง ๆ |
| `experiments/baseline` | โค้ดและผลการทดลองของระบบเดิมที่ใช้เป็น Baseline |
| `experiments/bge_m3` | การทดลองที่เกี่ยวข้องกับ BGE-M3 |
| `experiments/reranker` | การทดลองที่เกี่ยวข้องกับ Reranker |
| `results` | ผลลัพธ์จากการทดลองและการประเมินระบบ |
| `docs` | Paper, Dataset, Methodology และบันทึกการทำงานของโปรเจกต์ |
| `notebooks` | Jupyter Notebook สำหรับทดลอง วิเคราะห์ข้อมูล และทดสอบแนวคิด |
