from collections.abc import Sequence

from sentence_transformers import CrossEncoder

from .models import SearchResult


class CrossEncoderReranker:
    def __init__(self, model_name: str, batch_size: int = 8):
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        results: Sequence[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        pairs = [[query, result.document.context] for result in results]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        reranked = [
            SearchResult(
                result.document,
                float(score),
                result.bm25_score,
                result.semantic_score,
            )
            for result, score in zip(results, scores)
        ]
        return sorted(reranked, key=lambda result: result.score, reverse=True)[:top_k]

    def score_pairs(self, pairs):
        return self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=True,
        )