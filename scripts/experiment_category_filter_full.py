import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_qa import HybridRetriever, load_documents
from legal_qa.reranking import CrossEncoderReranker


MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
ALPHA = 0.5
CANDIDATE_K = 20
TOP_K = 5
KS = (1, 3, 5)


def selected_category(row, document_by_key):
    """Simulate a user-selected category without using it during retrieval."""
    positive_contexts = row.get("positive_contexts")
    if positive_contexts is None:
        return None, "no_positive_contexts"

    positive_keys = [str(item["unique_key"]) for item in positive_contexts]
    if not positive_keys:
        return None, "no_positive_contexts"

    categories = Counter(
        document_by_key[key].category
        for key in positive_keys
        if key in document_by_key
    )
    if not categories:
        return None, "no_mapped_positive_contexts"

    category = categories.most_common(1)[0][0]
    source = (
        "ambiguous_majority_category_of_positive_contexts"
        if len(categories) > 1
        else "majority_category_of_positive_contexts"
    )
    return category, source


def rank_condition(retriever, reranker, question, category):
    """Run the existing retrieval and reranking pipeline with an optional mask."""
    candidates = retriever.search(
        question,
        top_k=CANDIDATE_K,
        category=category,
    )
    return reranker.rerank(question, candidates, top_k=TOP_K)


def f1_for_ranking(retrieved_keys, positive_keys, k):
    retrieved = set(retrieved_keys[:k])
    positive = set(positive_keys)
    tp = len(retrieved & positive)
    fp = len(retrieved - positive)
    fn = len(positive - retrieved)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def recall_for_ranking(retrieved_keys, positive_keys, k):
    retrieved = set(retrieved_keys[:k])
    positive = set(positive_keys)
    return len(retrieved & positive) / len(positive) if positive else 0.0


def mrr_for_ranking(retrieved_keys, positive_keys):
    for rank, key in enumerate(retrieved_keys, start=1):
        if key in positive_keys:
            return 1.0 / rank
    return 0.0


def evaluate_rows(rows, documents, retriever, reranker, document_by_key):
    output = []
    skipped = 0
    ambiguous = 0
    valid = 0
    category_counts = Counter()
    positive_category_counts = Counter()
    positive_contexts_by_question = []

    for index, row in enumerate(rows):
        positive_contexts = row.get("positive_contexts")
        if positive_contexts is None or len(positive_contexts) == 0:
            skipped += 1
            output.append(
                {
                    "question_id": index,
                    "question": row["question"],
                    "selected_category": "",
                    "category_source": "no_positive_contexts",
                    "positive_keys": [],
                    "positive_category_count": 0,
                    "positive_category_counts": "",
                    "multi_category": False,
                    "baseline_top1": "",
                    "baseline_top3": "",
                    "baseline_top5": "",
                    "filtered_top1": "",
                    "filtered_top3": "",
                    "filtered_top5": "",
                    "baseline_mrr": "",
                    "filtered_mrr": "",
                    "baseline_f1_at_1": "",
                    "baseline_f1_at_3": "",
                    "baseline_f1_at_5": "",
                    "filtered_f1_at_1": "",
                    "filtered_f1_at_3": "",
                    "filtered_f1_at_5": "",
                }
            )
            continue

        positive_keys = [str(item["unique_key"]) for item in positive_contexts]
        positive_keys = [key for key in positive_keys if key in document_by_key]
        if not positive_keys:
            skipped += 1
            output.append(
                {
                    "question_id": index,
                    "question": row["question"],
                    "selected_category": "",
                    "category_source": "no_mapped_positive_contexts",
                    "positive_keys": [],
                    "positive_category_count": 0,
                    "positive_category_counts": "",
                    "multi_category": False,
                    "baseline_top1": "",
                    "baseline_top3": "",
                    "baseline_top5": "",
                    "filtered_top1": "",
                    "filtered_top3": "",
                    "filtered_top5": "",
                    "baseline_mrr": "",
                    "filtered_mrr": "",
                    "baseline_f1_at_1": "",
                    "baseline_f1_at_3": "",
                    "baseline_f1_at_5": "",
                    "filtered_f1_at_1": "",
                    "filtered_f1_at_3": "",
                    "filtered_f1_at_5": "",
                }
            )
            continue

        positive_categories = Counter(
            document_by_key[key].category for key in positive_keys
        )
        positive_category_counts.update(positive_categories)
        category, category_source = selected_category(row, document_by_key)
        if category is None:
            skipped += 1
            output.append(
                {
                    "question_id": index,
                    "question": row["question"],
                    "selected_category": "",
                    "category_source": category_source,
                    "positive_keys": positive_keys,
                    "positive_category_count": len(positive_category_counts),
                    "positive_category_counts": json.dumps(
                        dict(sorted(positive_categories.items())), ensure_ascii=False
                    ),
                    "multi_category": len(positive_categories) > 1,
                    "baseline_top1": "",
                    "baseline_top3": "",
                    "baseline_top5": "",
                    "filtered_top1": "",
                    "filtered_top3": "",
                    "filtered_top5": "",
                    "baseline_mrr": "",
                    "filtered_mrr": "",
                    "baseline_f1_at_1": "",
                    "baseline_f1_at_3": "",
                    "baseline_f1_at_5": "",
                    "filtered_f1_at_1": "",
                    "filtered_f1_at_3": "",
                    "filtered_f1_at_5": "",
                }
            )
            continue

        if len(positive_categories) > 1:
            ambiguous += 1

        category_counts[category] += 1
        valid += 1
        positive_contexts_by_question.append(set(positive_keys))

        baseline = rank_condition(
            retriever,
            reranker,
            row["question"],
            "all",
        )
        filtered = rank_condition(
            retriever,
            reranker,
            row["question"],
            category,
        )
        baseline_keys = [result.document.unique_key for result in baseline]
        filtered_keys = [result.document.unique_key for result in filtered]

        baseline_metrics = {
            k: f1_for_ranking(baseline_keys, positive_keys, k)
            for k in KS
        }
        filtered_metrics = {
            k: f1_for_ranking(filtered_keys, positive_keys, k)
            for k in KS
        }
        baseline_recall = {
            k: recall_for_ranking(baseline_keys, positive_keys, k)
            for k in KS
        }
        filtered_recall = {
            k: recall_for_ranking(filtered_keys, positive_keys, k)
            for k in KS
        }

        output.append(
            {
                "question_id": index,
                "question": row["question"],
                "selected_category": category,
                "category_source": category_source,
                "positive_keys": positive_keys,
                "positive_category_count": len(positive_categories),
                "positive_category_counts": json.dumps(
                    dict(sorted(positive_categories.items())), ensure_ascii=False
                ),
                "multi_category": len(positive_categories) > 1,
                "baseline_top1": baseline_keys[0],
                "baseline_top3": "|".join(baseline_keys[:3]),
                "baseline_top5": "|".join(baseline_keys[:5]),
                "filtered_top1": filtered_keys[0],
                "filtered_top3": "|".join(filtered_keys[:3]),
                "filtered_top5": "|".join(filtered_keys[:5]),
                "baseline_mrr": mrr_for_ranking(baseline_keys, positive_keys),
                "filtered_mrr": mrr_for_ranking(filtered_keys, positive_keys),
                "baseline_f1_at_1": baseline_metrics[1][2],
                "baseline_f1_at_3": baseline_metrics[3][2],
                "baseline_f1_at_5": baseline_metrics[5][2],
                "filtered_f1_at_1": filtered_metrics[1][2],
                "filtered_f1_at_3": filtered_metrics[3][2],
                "filtered_f1_at_5": filtered_metrics[5][2],
                "baseline_recall_at_1": baseline_recall[1],
                "baseline_recall_at_3": baseline_recall[3],
                "baseline_recall_at_5": baseline_recall[5],
                "filtered_recall_at_1": filtered_recall[1],
                "filtered_recall_at_3": filtered_recall[3],
                "filtered_recall_at_5": filtered_recall[5],
            }
        )

    return output, {
        "total_questions": len(rows),
        "valid_questions": valid,
        "skipped_questions": skipped,
        "ambiguous_multi_category_questions": ambiguous,
        "category_distribution": dict(sorted(category_counts.items())),
    }


def summarize(rows, stats):
    valid_rows = [row for row in rows if row["selected_category"]]
    metrics = {
        "F1@1": ("baseline_f1_at_1", "filtered_f1_at_1"),
        "F1@3": ("baseline_f1_at_3", "filtered_f1_at_3"),
        "F1@5": ("baseline_f1_at_5", "filtered_f1_at_5"),
        "Recall@1": ("baseline_recall_at_1", "filtered_recall_at_1"),
        "Recall@3": ("baseline_recall_at_3", "filtered_recall_at_3"),
        "Recall@5": ("baseline_recall_at_5", "filtered_recall_at_5"),
        "MRR@5": ("baseline_mrr", "filtered_mrr"),
    }

    summary = []
    for metric, (baseline_column, filtered_column) in metrics.items():
        baseline = float(np.mean([row[baseline_column] for row in valid_rows]))
        filtered = float(np.mean([row[filtered_column] for row in valid_rows]))
        absolute = filtered - baseline
        relative = absolute / baseline * 100 if baseline else 0.0
        summary.append(
            {
                "metric": metric,
                "baseline": baseline,
                "category_filtered": filtered,
                "absolute_improvement": absolute,
                "relative_improvement_percent": relative,
            }
        )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Compare full-test Hybrid retrieval with and without a simulated user-selected category filter"
    )
    parser.add_argument(
        "--test",
        default="data/raw/test-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--documents",
        default="data/processed/legal_documents_categorized.csv",
    )
    parser.add_argument(
        "--embeddings",
        default="results/minilm_document_embeddings.pt",
    )
    parser.add_argument(
        "--row-output",
        default="results/category_filter_test_all.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="results/category_filter_test_summary.csv",
    )
    parser.add_argument(
        "--sanity-size",
        type=int,
        default=None,
        help="Run only the first N questions and report a sanity check; do not overwrite full outputs.",
    )
    args = parser.parse_args()

    test = pd.read_parquet(args.test)
    if args.sanity_size is not None:
        if not 1 <= args.sanity_size <= len(test):
            raise ValueError("--sanity-size must be between 1 and the test-set size")
        test = test.head(args.sanity_size)
        print(f"SANITY CHECK: {len(test)} questions")
    else:
        print(f"FULL TEST: {len(test)} questions")

    documents = load_documents(args.documents)
    document_by_key = {document.unique_key: document for document in documents}
    retriever = HybridRetriever(
        documents,
        alpha=ALPHA,
        embeddings_path=args.embeddings,
    )
    reranker = CrossEncoderReranker(MODEL)

    rows, stats = evaluate_rows(
        [row for _, row in test.iterrows()],
        documents,
        retriever,
        reranker,
        document_by_key,
    )
    summary = summarize(rows, stats)

    row_frame = pd.DataFrame(rows)
    summary_frame = pd.DataFrame(
        [
            {
                "metric": item["metric"],
                "baseline": item["baseline"],
                "category_filtered": item["category_filtered"],
                "absolute_improvement": item["absolute_improvement"],
                "relative_improvement_percent": item["relative_improvement_percent"],
                "selected_category": "",
            }
            for item in summary
        ]
    )
    summary_frame.insert(
        1,
        "number_of_questions",
        stats["total_questions"],
    )
    summary_frame.insert(
        2,
        "valid_questions",
        stats["valid_questions"],
    )
    summary_frame.insert(
        3,
        "skipped_questions",
        stats["skipped_questions"],
    )
    summary_frame.insert(
        4,
        "ambiguous_multi_category_questions",
        stats["ambiguous_multi_category_questions"],
    )
    summary_frame.insert(
        5,
        "category_distribution",
        json.dumps(stats["category_distribution"], ensure_ascii=False),
    )

    Path(args.row_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    row_frame.to_csv(args.row_output, index=False)
    summary_frame.to_csv(args.summary_output, index=False)

    print("\nCATEGORY FILTER STATISTICS")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("\nMETRIC SUMMARY")
    print(summary_frame.to_string(index=False))

    if args.sanity_size is not None:
        print(f"\nSaved sanity output: {args.row_output}")
        print(f"Saved sanity summary: {args.summary_output}")
    else:
        print(f"\nSaved row-level results: {args.row_output}")
        print(f"Saved summary: {args.summary_output}")


if __name__ == "__main__":
    main()
