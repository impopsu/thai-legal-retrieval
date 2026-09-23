import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_qa import BM25Retriever, HybridRetriever, evaluate_retrieval, load_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Thai legal retrieval")
    parser.add_argument(
        "--documents",
        default="data/processed/legal_documents.csv",
    )
    parser.add_argument(
        "--test",
        default="data/raw/test-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("bm25", "hybrid"),
        default=["bm25"],
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--embeddings",
        default="results/minilm_document_embeddings.pt",
    )
    parser.add_argument(
        "--output",
        default="results/retrieval_metrics.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = load_documents(args.documents)
    test = pd.read_parquet(args.test)
    results = []

    for method in args.methods:
        if method == "bm25":
            retriever = BM25Retriever(documents)
        else:
            retriever = HybridRetriever(
                documents,
                alpha=args.alpha,
                embeddings_path=args.embeddings,
            )

        metrics = evaluate_retrieval(
            retriever,
            (row for _, row in test.iterrows()),
        )
        results.append({"method": method, "alpha": args.alpha, **metrics})
        print(f"\n{method}")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nบันทึกผลแล้ว: {output_path}")


if __name__ == "__main__":
    main()