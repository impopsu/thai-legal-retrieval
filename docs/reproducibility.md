# Checklist สำหรับการทำซ้ำผลการทดลอง

## ชุดข้อมูล

- Dataset: WangchanX-Legal-ThaiCCL-RAG
- Source: https://huggingface.co/datasets/airesearch/WangchanX-Legal-ThaiCCL-RAG
- ไฟล์ train/test ภายในเครื่องเก็บไว้ใต้ `data/raw/`
- ไม่ใช้ official test split สำหรับการ tuning

## Retrieval

- รันคำสั่งจาก root ของ repository
- สร้าง validation data ด้วย `scripts/split_train_validation.py`
- เลือก alpha ด้วย `scripts/select_alpha.py`
- ทำ benchmark methods ด้วย `scripts/benchmark_retrievers.py`
- บันทึก model names, alpha, candidate-k, top-k, metrics และ runtime

## Final configuration

```text
Retriever: Hybrid MiniLM
Alpha: 0.5
Candidate-k: 20
Reranker: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
Evidence top-k: 5
```

## คำสั่ง

```bash
python scripts/split_train_validation.py
python scripts/select_alpha.py
python scripts/benchmark_retrievers.py --split validation
python scripts/run_web_demo.py
```

กำหนดให้ Git เพิกเฉยต่อ embeddings, datasets และ result files ที่สร้างขึ้นโดยตั้งใจ
ให้สร้างไฟล์เหล่านี้ใหม่ด้วยคำสั่งข้างต้นเมื่อตั้งค่าสภาพแวดล้อมใหม่