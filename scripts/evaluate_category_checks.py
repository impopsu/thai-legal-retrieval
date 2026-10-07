import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_qa import HybridRetriever, load_documents
from legal_qa.categories import CATEGORY_LABELS


def build_accuracy_rows(documents, sample_size=30, seed=42):
    rng = random.Random(seed)
    docs = list(documents)
    rows = []
    for category in sorted(CATEGORY_LABELS):
        if category == "all":
            continue
        category_docs = [doc for doc in docs if doc.category == category]
        sample = rng.sample(category_docs, k=min(sample_size, len(category_docs))) if category_docs else []
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


def build_category_filter_recall(rows, documents, category_list=None):
    retriever = HybridRetriever(documents, alpha=0.5, embeddings_path="results/minilm_document_embeddings.pt")
    category_list = category_list or ["all", "criminal", "tax", "company", "property"]
    output = []
    for setting in category_list:
        recalls = {"recall_at_1": 0.0, "recall_at_3": 0.0, "recall_at_5": 0.0}
        num_questions = 0
        for row in rows:
            question = row["question"]
            results = retriever.search(question, top_k=5, category=setting)
            ranking = [result.document.unique_key for result in results]
            positive_set = {str(item["unique_key"]) for item in (row.get("positive_contexts") or [])}
            if not positive_set:
                continue
            num_questions += 1
            for k in (1, 3, 5):
                retrieved = set(ranking[:k])
                tp = len(retrieved & positive_set)
                total_relevant = len(positive_set)
                recalls[f"recall_at_{k}"] += (tp / total_relevant) if total_relevant else 0.0
        if num_questions:
            recalls = {key: value / num_questions for key, value in recalls.items()}
        output.append({"setting": setting, **recalls, "questions": num_questions})
    return pd.DataFrame(output)


def main():
    parser = argparse.ArgumentParser(description="Evaluate category accuracy and category filtering on the legal test set")
    parser.add_argument("--documents", default="data/processed/legal_documents_categorized.csv")
    parser.add_argument("--questions", default="data/raw/test-00000-of-00001.parquet")
    parser.add_argument("--accuracy-output", default="results/category_accuracy_check.csv")
    parser.add_argument("--filter-output", default="results/category_filter_recall.csv")
    parser.add_argument("--checks", choices=("all", "accuracy", "filter"), default="all")
    args = parser.parse_args()

    documents = load_documents(args.documents)
    if args.checks in ("all", "accuracy"):
        accuracy_df = build_accuracy_rows(documents)
        accuracy_df.to_csv(args.accuracy_output, index=False)
        print(accuracy_df.head())
        print(f"Saved category review sample to {args.accuracy_output}")

    if args.checks in ("all", "filter"):
        rows = [row for _, row in pd.read_parquet(args.questions).iterrows()]
        filter_df = build_category_filter_recall(rows, documents)
        filter_df.to_csv(args.filter_output, index=False)
        print(filter_df.to_string(index=False))
        print(f"Saved category filter summary to {args.filter_output}")


if __name__ == "__main__":
    main()
