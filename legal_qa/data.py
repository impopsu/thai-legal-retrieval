from pathlib import Path

import pandas as pd

from .categories import classify_category
from .models import LegalDocument


REQUIRED_COLUMNS = {"unique_key", "law_code", "law_title", "section", "context"}


def load_documents(path: str | Path) -> list[LegalDocument]:
    dataframe = pd.read_csv(path, encoding="utf-8-sig")
    missing = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing document columns: {sorted(missing)}")
    return [
        LegalDocument(
            unique_key=str(row.unique_key),
            law_code=str(row.law_code),
            law_title=str(row.law_title),
            section=str(row.section),
            context=str(row.context),
            category=str(
                row.category
                if "category" in dataframe.columns and pd.notna(row.category)
                else classify_category(
                    law_code=str(row.law_code),
                    section=str(row.section),
                    law_title=str(row.law_title),
                )
            ),
        )
        for row in dataframe.itertuples(index=False)
    ]