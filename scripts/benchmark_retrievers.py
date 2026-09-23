import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_qa import BM25Retriever, HybridRetriever, SemanticRetriever, load_documents
from legal_qa.evaluation import (
    evaluate_hybrid_batch,
    evaluate_retrieval,
    evaluate_semantic_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare retrieval methods on one fixed split"
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("bm25", "minilm", "hybrid_minilm", "bge", "hybrid_bge"),
        default=["bm25", "minilm", "hybrid_minilm"],
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--documents", default="data/processed/legal_documents.csv")
    parser.add_argument(
        "--validation",
        default="data/processed/validation_retrieval.parquet",
    )
    parser.add_argument("--test", default="data/raw/test-00000-of-00001.parquet")
    parser.add_argument("--minilm-embeddings", default="results/minilm_document_embeddings.pt")
    parser.add_argument("--bge-embeddings", default="results/bge_m3_document_embeddings.pt")
    parser.add_argument("--bge-batch-size", type=int, default=4)
    parser.add_argument("--output", default="results/retrieval_benchmark.csv")
    args = parser.parse_args()

    documents = load_documents(args.documents)
    split_path = args.validation if args.split == "validation" else args.test
    rows = [row for _, row in pd.read_parquet(split_path).iterrows()]
    results = []

    for method in args.methods:
        if method == "bm25":
            retriever = BM25Retriever(documents)
            metrics = evaluate_retrieval(retriever, rows)
        elif method == "minilm":
            retriever = SemanticRetriever(
                documents,
                embeddings_path=args.minilm_embeddings,
            )
            metrics = evaluate_semantic_batch(retriever, rows)
        elif method == "hybrid_minilm":
            retriever = HybridRetriever(
                documents,
                alpha=args.alpha,
                embeddings_path=args.minilm_embeddings,
            )
            metrics = evaluate_hybrid_batch(retriever, rows, alpha=args.alpha)
        elif method == "bge":
            retriever = SemanticRetriever(
                documents,
                model_name="BAAI/bge-m3",
                embeddings_path=args.bge_embeddings,
                batch_size=args.bge_batch_size,
            )
            metrics = evaluate_semantic_batch(retriever, rows)
        else:
            retriever = HybridRetriever(
                documents,
                model_name="BAAI/bge-m3",
                embeddings_path=args.bge_embeddings,
                alpha=args.alpha,
                batch_size=args.bge_batch_size,
            )
            metrics = evaluate_hybrid_batch(retriever, rows, alpha=args.alpha)

        result = {
            "split": args.split,
            "method": method,
            "alpha": args.alpha,
            "num_questions": len(rows),
            **metrics,
        }
        results.append(result)
        print(method, metrics)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"บันทึกผลแล้ว: {output_path}")


if __name__ == "__main__":
    main()