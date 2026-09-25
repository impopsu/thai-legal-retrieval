# โครงร่างรายงาน

1. **นิยามปัญหา**: การตอบคำถามกฎหมายไทยโดยใช้หลักฐานที่มีการกำกับแหล่งที่มา
2. **ชุดข้อมูล**: WangchanX-Legal-ThaiCCL-RAG, fields, split และข้อจำกัด
3. **งานวิจัยที่เกี่ยวข้อง**: NitiBench และระบบ legal QA แบบ free-format
4. **Methods**: BM25, MiniLM, BGE-M3, hybrid scoring และ cross-encoder reranking
5. **โพรโทคอลการประเมิน**: การปรับค่าใน validation และการรายงานผลจาก untouched test
6. **ผลลัพธ์**: retrieval benchmark, การปรับปรุงจาก reranker, runtime และ memory
7. **Grounded QA pipeline**: การเลือก evidence, ข้อจำกัดของ prompt และ citation
8. **การประเมินโดยมนุษย์**: เกณฑ์ correctness, faithfulness, citation และ abstention
9. **การวิเคราะห์ข้อผิดพลาด**: คำถามไม่เป็นทางการ, evidence ที่อ่อน และ corpus coverage
10. **ข้อจำกัด**: ยังไม่ได้เลือก LLM generator, ค่าใช้จ่ายบน CPU และ corpus ที่สร้างจาก benchmark
11. **บทสรุป**: configuration ที่เลือกและ trade-offs ที่สังเกตได้