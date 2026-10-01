from collections.abc import Iterable, Sequence


def evaluate_context_results(
    rows: Iterable,
    result_lists: Iterable[Sequence],
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    rows = list(rows)
    result_lists = list(result_lists)
    if not rows:
        raise ValueError("rows must contain at least one question")

    recall = {k: 0 for k in ks}
    precision_at_max = 0.0
    reciprocal_rank = 0.0
    for row, results in zip(rows, result_lists):
        positive_keys = {
            str(item["unique_key"])
            for item in row["positive_contexts"]
        }
        retrieved_keys = [result.document.unique_key for result in results]
        for k in ks:
            if positive_keys.intersection(retrieved_keys[:k]):
                recall[k] += 1
        hits = len(positive_keys.intersection(retrieved_keys[:max(ks)]))
        precision_at_max += hits / max(ks)
        for rank, key in enumerate(retrieved_keys[:max(ks)], start=1):
            if key in positive_keys:
                reciprocal_rank += 1 / rank
                break

    total = len(rows)
    return {
        **{f"ContextRecall@{k}": count / total for k, count in recall.items()},
        f"ContextPrecision@{max(ks)}": precision_at_max / total,
        f"ContextMRR@{max(ks)}": reciprocal_rank / total,
    }