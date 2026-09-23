import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_qa import HybridRetriever, load_documents


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select hybrid alpha on validation only"
    )
    parser.add_argument("--documents", default="data/processed/legal_documents.csv")
    parser.add_argument(
        "--validation",
        default="data/processed/validation_retrieval.parquet",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75],
    )
    parser.add_argument(
        "--embeddings",
        default="results/minilm_document_embeddings.pt",
    )
    parser.add_argument(
        "--output",
        default="results/validation_alpha_results.csv",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    documents = load_documents(args.documents)
    validation = pd.read_parquet(args.validation)
    if args.limit > 0:
        validation = validation.head(args.limit)
    rows = [row for _, row in validation.iterrows()]
    questions = validation["question"].fillna("").tolist()
    document_indices = {
        document.unique_key: index
        for index, document in enumerate(documents)
    }
    positive_indices = [
        [document_indices[str(item["unique_key"])] for item in row["positive_contexts"]]
        for row in rows
    ]
    results = []
    retriever = HybridRetriever(
        documents,
        alpha=args.alphas[0],
        embeddings_path=args.embeddings,
    )

    bm25_scores, semantic_scores = retriever.score_many(questions)

    for alpha in args.alphas:
        hybrid_scores = alpha * bm25_scores + (1 - alpha) * semantic_scores
        rankings = hybrid_scores.argsort(axis=1)[:, ::-1]
        recall_counts = {k: 0 for k in (1, 3, 5)}
        reciprocal_rank_sum = 0.0

        for ranking, positives in zip(rankings, positive_indices):
            positive_set = set(positives)
            for k in recall_counts:
                if positive_set.intersection(ranking[:k]):
                    recall_counts[k] += 1
            for rank, index in enumerate(ranking[:5], start=1):
                if index in positive_set:
                    reciprocal_rank_sum += 1 / rank
                    break

        total = len(rows)
        metrics = {
            **{f"Recall@{k}": count / total for k, count in recall_counts.items()},
            "MRR@5": reciprocal_rank_sum / total,
        }
        results.append({"alpha": alpha, **metrics})
        print(f"alpha={alpha}: {metrics}")

    dataframe = pd.DataFrame(results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    best = dataframe.sort_values("MRR@5", ascending=False).iloc[0]
    print(f"\nBest validation alpha: {best['alpha']}")
    print(f"บันทึกผลแล้ว: {output_path}")


if __name__ == "__main__":
    main()