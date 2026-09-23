import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_qa import BM25Retriever, HybridRetriever, load_documents
from legal_qa.evaluation import (
    evaluate_hybrid_batch,
    evaluate_retrieval,
    evaluate_reranker_batch,
)
from legal_qa.reranking import CrossEncoderReranker


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a cross-encoder reranker on validation data"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--documents", default="data/processed/legal_documents.csv")
    parser.add_argument(
        "--validation",
        default="data/processed/validation_retrieval.parquet",
    )
    parser.add_argument("--retriever", choices=("bm25", "hybrid"), default="hybrid")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--embeddings",
        default="results/minilm_document_embeddings.pt",
    )
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output",
        default="results/reranker_validation_results.csv",
    )
    args = parser.parse_args()

    documents = load_documents(args.documents)
    validation = pd.read_parquet(args.validation)
    if args.limit > 0:
        validation = validation.head(args.limit)
    rows = [row for _, row in validation.iterrows()]

    if args.retriever == "bm25":
        retriever = BM25Retriever(documents)
    else:
        retriever = HybridRetriever(
            documents,
            alpha=args.alpha,
            embeddings_path=args.embeddings,
        )

    reranker = CrossEncoderReranker(args.model, batch_size=args.batch_size)
    if args.retriever == "hybrid":
        baseline_metrics = evaluate_hybrid_batch(
            retriever,
            rows,
            alpha=args.alpha,
        )
    else:
        baseline_metrics = evaluate_retrieval(retriever, rows)
    metrics = evaluate_reranker_batch(
        retriever,
        reranker,
        rows,
        candidate_k=args.candidate_k,
        top_k=args.top_k,
        query_batch_size=args.query_batch_size,
    )
    print("baseline:", baseline_metrics)
    print("reranked:", metrics)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "model": args.model,
        "retriever": args.retriever,
        "alpha": args.alpha,
        "candidate_k": args.candidate_k,
        "top_k": args.top_k,
        **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
        **metrics,
    }]).to_csv(output_path, index=False)
    print(f"บันทึกผลแล้ว: {output_path}")


if __name__ == "__main__":
    main()