from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from pythainlp.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

from .models import LegalDocument, SearchResult


def tokenize(text: str) -> list[str]:
    return word_tokenize(str(text), engine="newmm")


def normalize(scores: np.ndarray) -> np.ndarray:
    minimum, maximum = scores.min(), scores.max()
    if minimum == maximum:
        return np.zeros_like(scores, dtype=float)
    return (scores - minimum) / (maximum - minimum)


class BM25Retriever:
    def __init__(self, documents: Sequence[LegalDocument]):
        self.documents = list(documents)
        self.index = BM25Okapi([tokenize(doc.context) for doc in self.documents])

    def score(self, query: str) -> np.ndarray:
        return np.asarray(self.index.get_scores(tokenize(query)), dtype=float)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        scores = self.score(query)
        indices = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(self.documents[i], float(scores[i])) for i in indices]


class SemanticRetriever:
    def __init__(
        self,
        documents: Sequence[LegalDocument],
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        embeddings_path: str | Path | None = None,
        batch_size: int = 32,
    ):
        self.documents = list(documents)
        self.model = SentenceTransformer(model_name)
        self.embeddings_path = Path(embeddings_path) if embeddings_path else None
        self.batch_size = batch_size
        self.document_embeddings = self._load_or_encode()

    def _load_or_encode(self) -> torch.Tensor:
        if self.embeddings_path and self.embeddings_path.exists():
            embeddings = torch.load(self.embeddings_path, weights_only=True)
            if embeddings.shape[0] == len(self.documents):
                return embeddings.to(self.model.device)
        embeddings = self.model.encode(
            [doc.context for doc in self.documents],
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        if self.embeddings_path:
            self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(embeddings.cpu(), self.embeddings_path)
        return embeddings

    def score(self, query: str) -> np.ndarray:
        query_embedding = self.model.encode(
            [query],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return (query_embedding @ self.document_embeddings.T)[0].detach().cpu().numpy()

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        scores = self.score(query)
        indices = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(self.documents[i], float(scores[i])) for i in indices]


class HybridRetriever:
    """alpha is the BM25 weight; semantic weight is 1 - alpha."""

    def __init__(
        self,
        documents: Sequence[LegalDocument],
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        embeddings_path: str | Path | None = None,
        alpha: float = 0.5,
        batch_size: int = 32,
    ):
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        self.documents = list(documents)
        self.alpha = alpha
        self.bm25 = BM25Retriever(self.documents)
        self.semantic = SemanticRetriever(
            self.documents,
            model_name=model_name,
            embeddings_path=embeddings_path,
            batch_size=batch_size,
        )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        bm25_scores = normalize(self.bm25.score(query))
        semantic_scores = normalize(self.semantic.score(query))
        hybrid_scores = self.alpha * bm25_scores + (1 - self.alpha) * semantic_scores
        indices = np.argsort(hybrid_scores)[::-1][:top_k]
        return [
            SearchResult(
                self.documents[i],
                float(hybrid_scores[i]),
                bm25_score=float(bm25_scores[i]),
                semantic_score=float(semantic_scores[i]),
            )
            for i in indices
        ]