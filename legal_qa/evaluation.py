from collections.abc import Iterable

import pandas as pd


def evaluate_retrieval(retriever, questions: Iterable[pd.Series], ks=(1, 3, 5)):
    rows = list(questions)
    if not rows:
        raise ValueError("questions must contain at least one row")
    max_k = max(ks)
    recall_counts = {k: 0 for k in ks}
    reciprocal_rank_sum = 0.0
    for row in rows:
        positive_keys = {str(item["unique_key"]) for item in row["positive_contexts"]}
        results = retriever.search(row["question"], top_k=max_k)
        ranked_keys = [result.document.unique_key for result in results]
        for k in ks:
            if positive_keys.intersection(ranked_keys[:k]):
                recall_counts[k] += 1
        for rank, key in enumerate(ranked_keys, start=1):
            if key in positive_keys:
                reciprocal_rank_sum += 1 / rank
                break
    total = len(rows)
    return {
        **{f"Recall@{k}": recall_counts[k] / total for k in ks},
        f"MRR@{max_k}": reciprocal_rank_sum / total,
    }