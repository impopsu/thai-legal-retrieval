# ชุดข้อมูล

โครงการนี้ใช้ **WangchanX-Legal-ThaiCCL-RAG**

- แหล่งที่มา: [Hugging Face: airesearch/WangchanX-Legal-ThaiCCL-RAG](https://huggingface.co/datasets/airesearch/WangchanX-Legal-ThaiCCL-RAG)
- ภาษา: Thai
- งาน: การตอบคำถามกฎหมายไทยและการสร้างแบบ Retrieval-Augmented Generation
- แหล่งข้อมูลหลักภายในเครื่อง: `data/raw/train-00000-of-00001.parquet` และ `data/raw/test-00000-of-00001.parquet`
- จำนวนตัวอย่าง Train ในไฟล์ปัจจุบัน: 8,211
- จำนวนตัวอย่าง Test ในไฟล์ปัจจุบัน: 3,742

แต่ละระเบียนประกอบด้วย:

```text
question
positive_contexts
hard_negative_contexts
positive_answer
hard_negative_answer
```

ดัชนีเอกสารที่ผ่านการประมวลผลในปัจจุบันสร้างไว้ใน
`data/processed/legal_documents.csv` จากระเบียน context ควรอธิบายการสร้างดัชนีนี้
ไว้อย่างชัดเจนในรายงาน เนื่องจาก label ของ benchmark เป็นแหล่ง context ที่ใช้สร้าง
ดัชนีภายในเครื่องนี้

## โพรโทคอลการประเมิน

- ไฟล์ Parquet ของ test อย่างเป็นทางการจะไม่ถูกแก้ไข
- สามารถแบ่งเฉพาะ training split ออกเป็น train และ validation subsets ได้
- ต้องเลือก model, alpha, top-k และการตั้งค่า reranking จาก validation
- ใช้ official test split เพียงครั้งเดียวสำหรับการรายงานผลสุดท้าย

## งานวิจัยที่เกี่ยวข้อง

1. Akarajaradwong et al. **NitiBench: Benchmarking LLM Frameworks on Thai
   Legal Question Answering Capabilities.** EMNLP 2025.
   [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1739/)
2. **A Free Format Legal Question Answering System.** NLLP 2021.
   [ACL Anthology](https://aclanthology.org/2021.nllp-1.11/)