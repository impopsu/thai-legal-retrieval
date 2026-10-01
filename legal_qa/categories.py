from collections.abc import Mapping


CATEGORY_KEYWORDS: Mapping[str, tuple[str, ...]] = {
    "murder": ("ฆ่า", "ฆาตกรรม", "ทำให้ผู้อื่นถึงแก่ความตาย", "ชีวิตและร่างกาย"),
    "property": (
        "ทรัพย์", "ที่ดิน", "เช่า", "ซื้อขาย", "จำนอง", "จำนำ", "กรรมสิทธิ์",
        "ลักทรัพย์", "ยักยอก", "ฉ้อโกง", "บุกรุก",
    ),
    "criminal": ("อาญา", "ความผิด", "โทษ", "ทำร้าย", "ปล้น", "โจรกรรม"),
    "financial": ("สถาบันการเงิน", "ธนาคาร", "การเงิน", "หลักทรัพย์", "เครดิต"),
    "company": ("บริษัท", "หุ้น", "นิติบุคคล", "ห้างหุ้นส่วน"),
    "labor": ("แรงงาน", "ลูกจ้าง", "นายจ้าง", "ค่าจ้าง", "เลิกจ้าง"),
    "family": ("ครอบครัว", "สมรส", "หย่า", "บุตร", "มรดก"),
    "tax": ("ภาษี", "อากร", "สรรพากร"),
    "procedure": ("วิธีพิจารณา", "ฟ้อง", "ศาล", "พยาน", "อุทธรณ์", "ฎีกา"),
}

CATEGORY_LABELS: Mapping[str, str] = {
    "all": "ทุกประเภท",
    "murder": "กฎหมายฆาตกรรม/ชีวิตและร่างกาย",
    "property": "กฎหมายทรัพย์สิน",
    "criminal": "กฎหมายอาญา",
    "financial": "กฎหมายการเงิน",
    "company": "กฎหมายบริษัทและนิติบุคคล",
    "labor": "กฎหมายแรงงาน",
    "family": "กฎหมายครอบครัวและมรดก",
    "tax": "กฎหมายภาษีอากร",
    "procedure": "กฎหมายวิธีพิจารณา",
    "other": "กฎหมายหมวดอื่น",
}


def classify_category(*texts: str) -> str:
    text = " ".join(str(value or "") for value in texts).lower()
    scores = {
        category: sum(text.count(keyword.lower()) for keyword in keywords)
        for category, keywords in CATEGORY_KEYWORDS.items()
    }
    best_category, best_score = max(scores.items(), key=lambda item: item[1])
    return best_category if best_score else "other"