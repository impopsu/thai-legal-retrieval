from collections.abc import Iterable

import numpy as np
import pandas as pd

from .retrieval import normalize_rows


def metrics_from_rankings(rankings, positive_indices, ks=(1, 3, 5)):
    recall_counts = {k: 0 for k in ks}
    reciprocal_rank_sum = 0.0
    for ranking, positives in zip(rankings, positive_indices):
        positive_set = set(positives)
        for k in ks:
            if positive_set.intersection(ranking[:k]):
                recall_counts[k] += 1
        for rank, index in enumerate(ranking[:max(ks)], start=1):
            if index in positive_set:
                reciprocal_rank_sum += 1 / rank
                break
    total = len(positive_indices)
    return {
        **{f"Recall@{k}": count / total for k, count in recall_counts.items()},
        f"MRR@{max(ks)}": reciprocal_rank_sum / total,
    }


def _positive_indices(retriever, rows):
    document_indices = {
        document.unique_key: index
        for index, document in enumerate(retriever.documents)
    }
    return [
        [document_indices[str(item["unique_key"])] for item in row["positive_contexts"]]
        for row in rows
    ]


def evaluate_score_matrix(retriever, rows, scores, ks=(1, 3, 5)):
    rankings = scores.argsort(axis=1)[:, ::-1]
    return metrics_from_rankings(
        rankings,
        _positive_indices(retriever, rows),
        ks=ks,
    )


def evaluate_semantic_batch(retriever, questions, ks=(1, 3, 5)):
    rows = list(questions)
    scores = retriever.score_many([row["question"] for row in rows])
    return evaluate_score_matrix(retriever, rows, scores, ks=ks)


def evaluate_hybrid_batch(retriever, questions, alpha, ks=(1, 3, 5)):
    rows = list(questions)
    bm25_scores, semantic_scores = retriever.score_many(
        [row["question"] for row in rows]
    )
    hybrid_scores = alpha * bm25_scores + (1 - alpha) * semantic_scores
    return evaluate_score_matrix(retriever, rows, hybrid_scores, ks=ks)


def evaluate_reranker(
    retriever,
    reranker,
    questions: Iterable[pd.Series],
    candidate_k: int = 20,
    top_k: int = 5,
    ks=(1, 3, 5),
):
    rows = list(questions)
    document_indices = {
        document.unique_key: index
        for index, document in enumerate(retriever.documents)
    }
    positive_indices = [
        [document_indices[str(item["unique_key"])] for item in row["positive_contexts"]]
        for row in rows
    ]
    rankings = []
    for row in rows:
        candidates = retriever.search(row["question"], top_k=candidate_k)
        reranked = reranker.rerank(row["question"], candidates, top_k=top_k)
        rankings.append([
            document_indices[result.document.unique_key]
            for result in reranked
        ])
    return metrics_from_rankings(rankings, positive_indices, ks=ks)


def evaluate_reranker_batch(
    retriever,
    reranker,
    questions: Iterable[pd.Series],
    candidate_k: int = 20,
    top_k: int = 5,
    query_batch_size: int = 32,
    ks=(1, 3, 5),
):
    rows = list(questions)
    queries = [row["question"] for row in rows]
    rankings = []
    for start in range(0, len(rows), query_batch_size):
        batch_rows = rows[start:start + query_batch_size]
        batch_queries = queries[start:start + query_batch_size]
        if hasattr(retriever, "semantic"):
            bm25_scores, semantic_scores = retriever.score_many(batch_queries)
            scores = retriever.alpha * bm25_scores + (1 - retriever.alpha) * semantic_scores
        else:
            scores = normalize_rows(retriever.score_many(batch_queries))

        candidate_indices = np.argsort(scores, axis=1)[:, ::-1][:, :candidate_k]
        pairs = [
            [batch_queries[row_index], retriever.documents[document_index].context]
            for row_index in range(len(batch_rows))
            for document_index in candidate_indices[row_index]
        ]
        reranker_scores = np.asarray(reranker.score_pairs(pairs)).reshape(
            len(batch_rows),
            candidate_k,
        )
        for row_index in range(len(batch_rows)):
            order = np.argsort(reranker_scores[row_index])[::-1][:top_k]
            rankings.append(candidate_indices[row_index][order])

    document_indices = {
        document.unique_key: index
        for index, document in enumerate(retriever.documents)
    }
    positive_indices = [
        [document_indices[str(item["unique_key"])] for item in row["positive_contexts"]]
        for row in rows
    ]
    return metrics_from_rankings(rankings, positive_indices, ks=ks)


def evaluate_retrieval(retriever, questions: Iterable[pd.Series], ks=(1, 3, 5)):
    rows = list(questions)
    if not rows:
        raise ValueError("questions must contain at least one row")
    max_k = max(ks)
    document_indices = {
        document.unique_key: index
        for index, document in enumerate(retriever.documents)
    }
    positive_indices = []
    rankings = []
    for row in rows:
        positive_indices.append([
            document_indices[str(item["unique_key"])]
            for item in row["positive_contexts"]
        ])
        results = retriever.search(row["question"], top_k=max_k)
        rankings.append([
            document_indices[result.document.unique_key]
            for result in results
        ])
    return metrics_from_rankings(rankings, positive_indices, ks=ks)