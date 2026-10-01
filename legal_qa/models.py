from dataclasses import dataclass


@dataclass(frozen=True)
class LegalDocument:
    unique_key: str
    law_code: str
    law_title: str
    section: str
    context: str
    category: str = "other"


@dataclass(frozen=True)
class SearchResult:
    document: LegalDocument
    score: float
    bm25_score: float | None = None
    semantic_score: float | None = None