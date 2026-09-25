# โพรโทคอลการประเมินโดยมนุษย์

metrics การ retrieval แบบอัตโนมัติไม่สามารถยืนยันได้ว่าคำตอบกฎหมายไทยที่สร้างขึ้น
มีประโยชน์หรือปลอดภัย ให้ใช้โพรโทคอลนี้หลังจากสร้างคำตอบจาก grounded prompts แล้ว

## การสุ่มตัวอย่าง

- สุ่มคำถาม 50-100 ข้อจาก untouched test set หลังจากล็อกระบบสุดท้ายแล้ว
- ห้ามใช้ผลการตัดสินเหล่านี้เพื่อเปลี่ยน alpha, top-k, model หรือ reranker
- หากเป็นไปได้ ให้ผู้ประเมินอย่างน้อยสองคนให้คะแนนคำตอบแต่ละข้อ

## เกณฑ์การให้คะแนน

ให้คะแนนแต่ละรายการตั้งแต่ 0 ถึง 2:

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| Answer correctness | Wrong or misleading | Partly correct | Correct |
| Evidence faithfulness | Unsupported by context | Partly supported | Fully supported |
| Citation correctness | Wrong/missing | Partly correct | Correct law and section |
| Completeness | Misses the main issue | Partly addresses it | Addresses the question |
| Abstention behavior | Hallucinates when evidence is weak | Unclear | Correctly refuses or qualifies |

คอลัมน์ที่แนะนำสำหรับการทำ annotation:

```text
question
answer
retrieved_sources
answer_correctness
evidence_faithfulness
citation_correctness
completeness
abstention_behavior
notes
reviewer_id
```

รายงานค่าเฉลี่ยของแต่ละเกณฑ์ ความสอดคล้องระหว่างผู้ประเมิน และตัวอย่างทั้งคำตอบ
ที่ประสบความสำเร็จและคำตอบที่ไม่สำเร็จ