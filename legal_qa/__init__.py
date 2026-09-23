"""Small, reusable core for the Thai Legal QA project."""

from .data import load_documents
from .evaluation import evaluate_retrieval
from .models import LegalDocument, SearchResult
from .qa import LegalQAPipeline
from .retrieval import BM25Retriever, HybridRetriever, SemanticRetriever

__all__ = [
    "BM25Retriever",
    "HybridRetriever",
    "LegalDocument",
    "LegalQAPipeline",
    "SearchResult",
    "SemanticRetriever",
    "evaluate_retrieval",
    "load_documents",
]