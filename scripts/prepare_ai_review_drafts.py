import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_qa.categories import classify_category


ANSWER_REVIEW = {
    "validation": [
        ({"correctness": 5, "faithfulness": 5, "citation": 5, "completeness": 4, "abstention": 5}, "ตรงกับเฉลยและมาตรา 18 ใน retrieved sources; คำตอบกระชับพอสำหรับคำถาม"),
        ({"correctness": 2, "faithfulness": 2, "citation": 2, "completeness": 2, "abstention": 3}, "กล่าวอ้างสิทธิในสินทรัพย์กว้างกว่าหลักฐานมาตรา 15/1, 16 และ 19 ที่แสดง; ควรจำกัดคำตอบตามสิทธิที่บทบัญญัติระบุ"),
        ({"correctness": 4, "faithfulness": 4, "citation": 4, "completeness": 4, "abstention": 5}, "หลักการมอบฉันทะตรงกับมาตรา 1187 และ 102; รายละเอียดแบบหนังสือช่วยตอบ แต่ควรตรวจเลขมาตราที่อ้างในข้อความ"),
        ({"correctness": 3, "faithfulness": 4, "citation": 2, "completeness": 2, "abstention": 4}, "งดตอบเมื่อหลักฐานที่ค้นได้ไม่ยืนยันเรื่องจดทะเบียน; แหล่งข้อมูลเกี่ยวกับการเปลี่ยน trustee แต่ยังไม่ตอบประเด็น registration"),
        ({"correctness": 4, "faithfulness": 5, "citation": 1, "completeness": 2, "abstention": 5}, "งดตอบเหมาะสมเพราะ retrieved sources เป็นกฎหมายสัญญาซื้อขายล่วงหน้า ไม่ใช่ข้อยกเว้นใบอนุญาตตามกฎหมายหลักทรัพย์"),
        ({"correctness": 5, "faithfulness": 5, "citation": 5, "completeness": 4, "abstention": 5}, "ตอบหลักของมาตรา 393 และมีแหล่งกฎหมายตรงคำถาม; สำนวนขยายบางส่วนแต่ไม่เปลี่ยนสาระ"),
        ({"correctness": 2, "faithfulness": 4, "citation": 1, "completeness": 1, "abstention": 2}, "งดตอบทั้งที่ retrieved sources มีมาตรา 412, 414 และ 418 เกี่ยวกับการคืนทรัพย์; ควรสรุปหลักและเงื่อนไขจากหลักฐาน"),
        ({"correctness": 3, "faithfulness": 5, "citation": 1, "completeness": 1, "abstention": 4}, "การงดตอบปลอดภัย เพราะมาตรา 20 กล่าวถึงการจัดทำบัญชี ส่วนมาตรา 174 กล่าวถึงผู้ชำระบัญชี ไม่ใช่หน้าที่ผู้ทำบัญชีโดยตรง"),
        ({"correctness": 5, "faithfulness": 4, "citation": 4, "completeness": 4, "abstention": 5}, "ตรงกับรายการงบประมาณในมาตรา 56 ที่ถูกค้นพบ; ควรตรวจว่ารายการย่อยทั้งหมดในคำตอบมีอยู่ครบตามบทบัญญัติ"),
        ({"correctness": 4, "faithfulness": 4, "citation": 4, "completeness": 3, "abstention": 5}, "จำนวนประเภทตรงกับเฉลยสองประเภท; คำอธิบายเพิ่มเรื่องประเภทวิชาการควรมีหลักฐานรองรับหรือเอาออก"),
        ({"correctness": 5, "faithfulness": 5, "citation": 5, "completeness": 4, "abstention": 5}, "สรุปเหตุระงับตามมาตรา 35 ได้ตรงกับแหล่งที่ค้นพบ; ควรระบุเงื่อนไข/ขั้นตอนครบตามข้อความกฎหมาย"),
        ({"correctness": 2, "faithfulness": 4, "citation": 1, "completeness": 1, "abstention": 2}, "งดตอบทั้งที่มาตรา 501 อยู่ใน retrieved sources และเกี่ยวกับสภาพทรัพย์ที่ไถ่; ควรอธิบายหน้าที่และข้อยกเว้นตามมาตรานั้น"),
        ({"correctness": 1, "faithfulness": 1, "citation": 1, "completeness": 2, "abstention": 1}, "ตัวเลขสัดส่วน 49% และชื่อกฎหมายไม่สอดคล้องกับมาตรา 15 ที่ค้นพบ ซึ่งกำหนดสัดส่วนผู้ถือหุ้นไทยและข้อยกเว้น"),
        ({"correctness": 5, "faithfulness": 5, "citation": 4, "completeness": 4, "abstention": 5}, "แยกอำนาจเชิญออกจากการบังคับเรียกได้เหมาะสม และมีมาตรา 15 รองรับ"),
        ({"correctness": 2, "faithfulness": 5, "citation": 1, "completeness": 1, "abstention": 2}, "งดตอบทั้งที่มาตรา 143 ซึ่งนิยามทรัพย์นอกพาณิชย์อยู่ใน retrieved sources; ควรตอบตามบทนิยามโดยไม่ขยายข้อสรุปเกินกฎหมาย"),
        ({"correctness": 3, "faithfulness": 5, "citation": 1, "completeness": 1, "abstention": 4}, "การงดตอบเหมาะกับหลักฐานที่คืนมา เพราะ retrieved sections กล่าวถึงข้อห้ามคนละเรื่องและไม่มีมาตรา 312 ที่อ้างในเฉลย"),
        ({"correctness": 5, "faithfulness": 5, "citation": 5, "completeness": 4, "abstention": 5}, "ตรงกับมาตรา 309 และ retrieved source; ควรคงเงื่อนไขเรื่องเจตนาก่อความเสียหายให้ชัด"),
        ({"correctness": 5, "faithfulness": 5, "citation": 5, "completeness": 5, "abstention": 5}, "แจกแจงสี่วิธีตรงกับมาตรา 79 ซึ่งเป็น retrieved source อันดับแรก"),
        ({"correctness": 4, "faithfulness": 5, "citation": 5, "completeness": 3, "abstention": 5}, "คำตอบตรงกับมาตรา 11 ใน retrieved sources แต่ไม่ตรงกับ reference answer ที่กล่าวถึงอีกกฎหมายหนึ่ง; ต้องตรวจความถูกต้องของ reference"),
        ({"correctness": 4, "faithfulness": 5, "citation": 4, "completeness": 4, "abstention": 5}, "สาระเรื่องประโยชน์สูงสุดและความซื่อสัตย์ตรงกับมาตรา 89/7 และ 89/10; คำตอบรวมหลักหลายข้อได้เหมาะสม"),
    ],
    "test": [
        ({"correctness": 5, "faithfulness": 5, "citation": 5, "completeness": 4, "abstention": 5}, "ตอบยืนยันตรงกับเฉลยและมาตรา 29/30 ที่ค้นพบ; คำถามต้องการคำตอบใช่/ไม่ใช่จึงไม่จำเป็นต้องขยายมาก"),
        ({"correctness": 4, "faithfulness": 5, "citation": 4, "completeness": 4, "abstention": 5}, "สรุปอำนาจธนาคารแห่งประเทศไทยตามมาตรา 95 ได้ แต่ควรตรวจว่ารายละเอียดคำสั่งทั้งหมดครบถ้วน"),
        ({"correctness": 5, "faithfulness": 5, "citation": 5, "completeness": 5, "abstention": 5}, "ตอบตรงและแสดงเหตุผลจากมาตรา 87 ซึ่งปรากฏเป็น retrieved source อันดับแรก"),
        ({"correctness": 5, "faithfulness": 5, "citation": 5, "completeness": 4, "abstention": 5}, "แนวทางแต่งตั้งผู้ตรวจการและสอบสวนสอดคล้องกับมาตรา 93; คำตอบควรรักษาลำดับขั้นตอนตามบทบัญญัติ"),
        ({"correctness": 5, "faithfulness": 5, "citation": 4, "completeness": 4, "abstention": 5}, "สรุปความรับผิดร่วมของสถาบันที่ควบ/รับโอนกิจการตรงกับเฉลย; ตรวจให้แน่ใจว่าข้อยกเว้นไม่ตกหล่น"),
        ({"correctness": 5, "faithfulness": 5, "citation": 4, "completeness": 4, "abstention": 5}, "สรุปอำนาจกำหนดอัตราส่วนของกลุ่มธุรกิจตามมาตรา 57 ได้ตรง แต่ควรระบุชนิดอัตราส่วนให้ครบ"),
        ({"correctness": 4, "faithfulness": 4, "citation": 3, "completeness": 3, "abstention": 5}, "ตอบว่าต้องดำรงสินทรัพย์/หลักทรัพย์ตามเกณฑ์ แต่คำถามถามต่อว่าถือเป็นเงินกองทุนหรือไม่; ต้องตรวจว่าคำตอบครอบคลุมส่วนหลัง"),
        ({"correctness": 5, "faithfulness": 5, "citation": 4, "completeness": 5, "abstention": 5}, "ปฏิเสธการกันสำรอง 10% สอดคล้องกับเพดานตามมาตรา 61 ที่ค้นพบ"),
        ({"correctness": 5, "faithfulness": 5, "citation": 5, "completeness": 4, "abstention": 5}, "คำตอบว่าให้รายงานโดยไม่ชักช้าตรงกับเฉลยและมาตรา 109/110 ที่อ้าง"),
        ({"correctness": 5, "faithfulness": 4, "citation": 4, "completeness": 4, "abstention": 5}, "ตอบว่ามีหน้าที่เปิดเผยและมีมาตรา 31/38 รองรับ; ควรแยกประเภทข้อมูลและผู้รับข้อมูลให้ชัด"),
        ({"correctness": 2, "faithfulness": 5, "citation": 1, "completeness": 1, "abstention": 3}, "งดตอบแต่เฉลยระบุว่ามีอำนาจควบคุม; retrieved sources ไม่แสดงบทนิยามที่ชัดเจน จึงควรค้นเพิ่มหรืออธิบายข้อจำกัด"),
        ({"correctness": 5, "faithfulness": 5, "citation": 5, "completeness": 4, "abstention": 5}, "ระบุให้แจ้งตลาดหลักทรัพย์เป็นหนังสือตรงกับมาตรา 101 ที่ค้นพบ"),
        ({"correctness": 5, "faithfulness": 5, "citation": 5, "completeness": 4, "abstention": 5}, "คำตอบเรื่องพ้นล้มละลายเกินห้าปีตรงกับมาตรา 24; คำถามให้ข้อมูลแปดปีจึงตอบได้ตรง"),
        ({"correctness": 5, "faithfulness": 5, "citation": 5, "completeness": 4, "abstention": 5}, "ตอบข้อห้ามเกินอัตราส่วนตามมาตรา 58 ซึ่งเป็น retrieved source อันดับแรก"),
        ({"correctness": 3, "faithfulness": 5, "citation": 1, "completeness": 1, "abstention": 5}, "การงดตอบเหมาะสมเมื่อ retrieved sources ไม่มีมาตรา 47 ที่เฉลยอ้าง; แหล่งที่คืนมาไม่รองรับคำตอบตรงประเด็น"),
        ({"correctness": 3, "faithfulness": 4, "citation": 2, "completeness": 1, "abstention": 3}, "งดตอบ แต่ retrieved source มาตรา 23 กล่าวถึงข้อยกเว้นที่เกี่ยวข้อง; ควรอธิบายว่าไม่ต้องขายเมื่อข้อยกเว้นใช้บังคับ"),
        ({"correctness": 2, "faithfulness": 4, "citation": 1, "completeness": 1, "abstention": 2}, "งดตอบทั้งที่มาตรา 31 เรื่องเปิดเผยความเสี่ยงอยู่ใน retrieved sources; ควรตอบพร้อมเงื่อนไขตามประกาศ ธปท."),
        ({"correctness": 5, "faithfulness": 5, "citation": 5, "completeness": 5, "abstention": 5}, "คำตอบตรงกับมาตรา 17 และข้อยกเว้นหุ้นบุริมสิทธิไม่มีสิทธิออกเสียง"),
        ({"correctness": 5, "faithfulness": 5, "citation": 5, "completeness": 4, "abstention": 5}, "สรุปข้อจำกัดการโอนหุ้นตามมาตรา 15 ได้ตรงและมีแหล่งกฎหมายตรงคำถาม"),
        ({"correctness": 5, "faithfulness": 5, "citation": 5, "completeness": 5, "abstention": 5}, "ยกเว้นธุรกรรมที่รัฐค้ำประกันตามมาตรา 52 ตรงกับเฉลยและ retrieved source"),
    ],
}


def prepare_answer_review(source_path, output_path):
    grouped = {"validation": [], "test": []}
    with Path(source_path).open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            grouped[row["split"]].append(row)

    for split, rows in grouped.items():
        if len(rows) != len(ANSWER_REVIEW[split]):
            raise ValueError(
                f"Expected {len(ANSWER_REVIEW[split])} {split} rows, got {len(rows)}"
            )
        for row, (scores, rationale) in zip(rows, ANSWER_REVIEW[split]):
            row["ai_review_draft"] = {
                "reviewer_id": "AI draft (GitHub Copilot)",
                "status": "provisional; requires human verification",
                "scores": scores,
                "rationale": rationale,
                "basis": "question, model answer, reference answer, and retrieved source texts",
            }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for split in ("validation", "test"):
            for row in grouped[split]:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_category_review(source_path, output_path):
    frame = pd.read_csv(source_path, encoding="utf-8-sig")
    frame["ai_suggested_category"] = [
        classify_category(
            law_code=row.law_code,
            section=row.section,
            law_title=row.law_title,
        )
        for row in frame.itertuples(index=False)
    ]
    frame["suggestion_method"] = "dataset-backed law_code + section mapping; provisional"
    frame["suggestion_matches_auto"] = (
        frame["ai_suggested_category"] == frame["auto_category"]
    ).astype(int)
    frame["review_status"] = "requires human verification"
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare provisional AI/rule-based review drafts without changing human forms"
    )
    parser.add_argument(
        "--answers-input",
        default="results/human_eval_answers_rubric.jsonl",
    )
    parser.add_argument(
        "--answers-output",
        default="results/human_eval_ai_review_draft.jsonl",
    )
    parser.add_argument(
        "--categories-input",
        default="results/category_accuracy_check.csv",
    )
    parser.add_argument(
        "--categories-output",
        default="results/category_ai_review_draft.csv",
    )
    args = parser.parse_args()

    prepare_answer_review(args.answers_input, args.answers_output)
    prepare_category_review(args.categories_input, args.categories_output)
    print(f"Saved provisional answer review: {args.answers_output}")
    print(f"Saved provisional category suggestions: {args.categories_output}")


if __name__ == "__main__":
    main()
