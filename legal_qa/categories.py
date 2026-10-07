from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import pandas as pd

# Deprecated compatibility constants retained for UI labels and older code paths.
# They are intentionally not used in the production category assignment pipeline.
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

_DEFAULT_CATEGORY_DATASET = Path(__file__).resolve().parents[1] / "data" / "processed" / "legal_documents_categorized.csv"
_DEFAULT_RAW_DATASET = Path(__file__).resolve().parents[1] / "data" / "processed" / "legal_documents.csv"


def normalize_section_key(section: str | int | None) -> str | None:
    if section is None:
        return None
    text = str(section).strip()
    if not text or text.lower() == "nan":
        return None
    match = re.search(r"\d+(?:/\d+)?(?:\s+[ก-ฮ]+)?", text)
    if not match:
        return None
    return match.group(0).strip()


@lru_cache(maxsize=None)
def _derive_dataset_reference(dataset_path: str | Path | None = None) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, str]]:
    dataset = Path(dataset_path) if dataset_path else _DEFAULT_CATEGORY_DATASET
    dataset = dataset.resolve()
    if not dataset.exists():
        fallback = _DEFAULT_RAW_DATASET.resolve()
        if not fallback.exists():
            return ({}, {}, {})
        dataset = fallback

    dataframe = pd.read_csv(dataset, encoding="utf-8-sig")
    if "law_code" not in dataframe.columns:
        return ({}, {}, {})

    law_to_section: dict[str, dict[str, Counter[str]]] = {}
    law_totals: dict[str, Counter[str]] = {}
    title_to_law_code: dict[str, str] = {}

    for _, row in dataframe.iterrows():
        law_code = str(row.get("law_code", "")).strip()
        if not law_code or law_code.lower() == "nan":
            continue
        law_title = str(row.get("law_title", "")).strip()
        if law_title:
            title_to_law_code.setdefault(law_title, law_code)

        category = str(row.get("category", "")).strip()
        if not category:
            category = "other"

        section_key = normalize_section_key(row.get("section"))
        if section_key is None:
            law_totals.setdefault(law_code, Counter())[category] += 1
            continue

        law_to_section.setdefault(law_code, {})
        law_to_section[law_code].setdefault(section_key, Counter())[category] += 1
        law_totals.setdefault(law_code, Counter())[category] += 1

    by_law: dict[str, dict[str, str]] = {}
    law_majority: dict[str, str] = {}
    for law_code, section_counts in law_to_section.items():
        by_law[law_code] = {
            section_key: counts.most_common(1)[0][0]
            for section_key, counts in section_counts.items()
        }
        law_majority[law_code] = law_totals.get(law_code, Counter()).most_common(1)[0][0]

    return by_law, law_majority, title_to_law_code


def classify_category(
    *texts: str,
    law_code: str | None = None,
    section: str | None = None,
    law_title: str | None = None,
    dataset_path: str | Path | None = None,
) -> str:
    """Classify a document using its dataset-backed law_code and article section.

    This does not inspect question text, context text, or semantic similarity.
    It resolves from actual corpus metadata only: law_code + section, then falls back
    to the law's dominant category when a section is missing.
    """
    by_law, law_majority, title_to_law_code = _derive_dataset_reference(dataset_path)

    if law_code is None and law_title is not None:
        law_code = title_to_law_code.get(str(law_title).strip())

    if law_code is None and texts:
        for value in texts:
            candidate = str(value or "").strip()
            if not candidate:
                continue
            if candidate in title_to_law_code:
                law_code = title_to_law_code[candidate]
                break
            if re.fullmatch(r"[A-Za-z0-9ก-๙\-]+", candidate) and "-" in candidate:
                law_code = candidate
                break

    law_code = str(law_code).strip() if law_code is not None else None
    if not law_code or law_code.lower() == "nan":
        return "other"

    section_key = normalize_section_key(section)
    if section_key is None and texts:
        for value in texts:
            candidate = normalize_section_key(value)
            if candidate is not None:
                section_key = candidate
                break

    if section_key is not None and law_code in by_law and section_key in by_law[law_code]:
        return by_law[law_code][section_key]
    if law_code in law_majority:
        return law_majority[law_code]
    if law_title and law_title in title_to_law_code:
        return classify_category(
            law_code=title_to_law_code[str(law_title).strip()],
            section=section,
            dataset_path=dataset_path,
        )
    return "other"


def rebuild_corpus_categories(
    source_path: str | Path = "data/processed/legal_documents.csv",
    output_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
) -> pd.DataFrame:
    source = Path(source_path)
    dataframe = pd.read_csv(source, encoding="utf-8-sig")
    dataframe["category"] = dataframe.apply(
        lambda row: classify_category(
            law_code=str(row.get("law_code", "")),
            section=str(row.get("section", "")),
            law_title=str(row.get("law_title", "")),
            dataset_path=dataset_path or _DEFAULT_CATEGORY_DATASET,
        ),
        axis=1,
    )
    target = Path(output_path) if output_path else source.with_name("legal_documents_categorized.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(target, index=False, encoding="utf-8-sig")
    return dataframe
