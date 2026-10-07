import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_qa import HybridRetriever, load_documents
from legal_qa.categories import CATEGORY_LABELS
from legal_qa.reranking import CrossEncoderReranker


FINAL_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def build_accuracy_rows(documents, sample_size=30, seed=42):
    rng = random.Random(seed)
    rows = []
    for category in sorted(CATEGORY_LABELS):
        if category == "all":
            continue
        category_docs = [doc for doc in documents if doc.category == category]
        sample = rng.sample(category_docs, k=min(sample_size, len(category_docs)))
        for doc in sample:
            rows.append({
                "unique_key": doc.unique_key,
                "law_code": doc.law_code,
                "law_title": doc.law_title,
                "section": doc.section,
                "context": doc.context,
                "auto_category": doc.category,
                "human_category": "",
                "correct": "",
                "reviewer_id": "",
                "notes": "",
            })
    return pd.DataFrame(rows, columns=[
        "unique_key",
        "law_code",
        "law_title",
        "section",
        "context",
        "auto_category",
        "human_category",
        "correct",
        "reviewer_id",
        "notes",
    ])


def build_category_filter_recall(
    rows,
    documents,
    embeddings_path="results/minilm_document_embeddings.pt",
    alpha=0.5,
    candidate_k=20,
    top_k=5,
):
    retriever = HybridRetriever(
        documents,
        alpha=alpha,
        embeddings_path=embeddings_path,
    )
    reranker = CrossEncoderReranker(FINAL_RERANKER_MODEL)
    document_by_key = {doc.unique_key: doc for doc in documents}
    recall_sums = {"unfiltered": 0.0, "oracle_category_filter": 0.0}
    skipped_no_ground_truth = 0
    skipped_missing_document = 0
    prepared = []

    for row in rows:
        contexts = row.get("positive_contexts")
        contexts = [] if contexts is None else list(contexts)
        positive_keys = {str(item["unique_key"]) for item in contexts}
        if not positive_keys:
            skipped_no_ground_truth += 1
            continue
        positive_set = positive_keys & document_by_key.keys()
        if not positive_set:
            skipped_missing_document += 1
            continue

        category_counts = Counter(
            document_by_key[key].category for key in positive_set
        )
        oracle_category = min(
            category_counts,
            key=lambda category: (-category_counts[category], category),
        )
        category_positive_set = {
            key
            for key in positive_set
            if document_by_key[key].category == oracle_category
        }
        prepared.append((row["question"], oracle_category, category_positive_set))

    valid_questions = len(prepared)
    if not valid_questions:
        raise ValueError("No questions have ground-truth documents in the corpus")

    query_batch_size = 16
    for batch_start in range(0, valid_questions, query_batch_size):
        batch = prepared[batch_start:batch_start + query_batch_size]
        queries = [item[0] for item in batch]
        bm25_scores, semantic_scores = retriever.score_many(queries)
        combined_scores = alpha * bm25_scores + (1 - alpha) * semantic_scores
        candidate_groups = []
        group_offsets = []
        pairs = []

        for row_index, (_, oracle_category, _) in enumerate(batch):
            scores = combined_scores[row_index]
            unfiltered_indices = np.argsort(scores)[::-1][:candidate_k]
            category_mask = np.fromiter(
                (doc.category == oracle_category for doc in documents),
                dtype=bool,
                count=len(documents),
            )
            filtered_scores = scores.copy()
            filtered_scores[~category_mask] = -np.inf
            filtered_indices = np.argsort(filtered_scores)[::-1][
                :min(candidate_k, int(category_mask.sum()))
            ]

            for indices in (unfiltered_indices, filtered_indices):
                group_offsets.append(len(pairs))
                candidate_groups.append(indices)
                pairs.extend([
                    [queries[row_index], documents[index].context]
                    for index in indices
                ])

        pair_scores = np.asarray(reranker.score_pairs(pairs)).reshape(-1)
        for row_index, (_, _, category_positive_set) in enumerate(batch):
            for setting_index, setting in enumerate(
                ("unfiltered", "oracle_category_filter")
            ):
                group_index = row_index * 2 + setting_index
                indices = candidate_groups[group_index]
                start_score = group_offsets[group_index]
                end_score = start_score + len(indices)
                order = np.argsort(pair_scores[start_score:end_score])[::-1][:top_k]
                ranked_keys = {
                    documents[indices[position]].unique_key for position in order
                }
                recall_sums[setting] += len(
                    ranked_keys & category_positive_set
                ) / len(category_positive_set)

    category_source = "majority category of mapped positive contexts"
    return pd.DataFrame([
        {
            "setting": setting,
            "category_source": category_source,
            "Recall@5": recall / valid_questions,
            "questions": valid_questions,
            "total_questions": len(rows),
            "skipped_no_ground_truth": skipped_no_ground_truth,
            "skipped_missing_document": skipped_missing_document,
        }
        for setting, recall in recall_sums.items()
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate category accuracy and category filtering"
    )
    parser.add_argument(
        "--documents",
        default="data/processed/legal_documents_categorized.csv",
    )
    parser.add_argument(
        "--questions",
        default="data/processed/validation_retrieval.parquet",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--embeddings",
        default="results/minilm_document_embeddings.pt",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--accuracy-output",
        default="results/category_accuracy_check.csv",
    )
    parser.add_argument(
        "--filter-output",
        default="results/category_filter_recall.csv",
    )
    parser.add_argument(
        "--checks",
        choices=("all", "accuracy", "filter"),
        default="all",
    )
    args = parser.parse_args()

    documents = load_documents(args.documents)
    if args.checks in ("all", "accuracy"):
        accuracy_df = build_accuracy_rows(documents)
        accuracy_df.to_csv(args.accuracy_output, index=False)
        print(f"Saved category review sample to {args.accuracy_output}")

    if args.checks in ("all", "filter"):
        questions = pd.read_parquet(args.questions).head(args.limit)
        rows = [row for _, row in questions.iterrows()]
        filter_df = build_category_filter_recall(
            rows,
            documents,
            embeddings_path=args.embeddings,
            alpha=args.alpha,
            candidate_k=args.candidate_k,
            top_k=args.top_k,
        )
        filter_df.to_csv(args.filter_output, index=False)
        print(filter_df.to_string(index=False))
        print(f"Saved category filter summary to {args.filter_output}")


if __name__ == "__main__":
    main()
