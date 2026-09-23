import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_qa import HybridRetriever, evaluate_retrieval, load_documents


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
    args = parser.parse_args()

    documents = load_documents(args.documents)
    validation = pd.read_parquet(args.validation)
    rows = [row for _, row in validation.iterrows()]
    results = []
    retriever = HybridRetriever(
        documents,
        alpha=args.alphas[0],
        embeddings_path=args.embeddings,
    )

    for alpha in args.alphas:
        retriever.alpha = alpha
        metrics = evaluate_retrieval(retriever, rows)
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