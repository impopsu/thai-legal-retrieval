import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_qa import HybridRetriever, LegalQAPipeline, load_documents
from legal_qa.categories import CATEGORY_LABELS
from legal_qa.context_evaluation import evaluate_context_results
from legal_qa.reranking import CrossEncoderReranker


FINAL_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieved contexts")
    parser.add_argument("--questions", default="data/processed/validation_retrieval.parquet")
    parser.add_argument("--documents", default="data/processed/legal_documents.csv")
    parser.add_argument("--embeddings", default="results/minilm_document_embeddings.pt")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--category", choices=tuple(CATEGORY_LABELS), default="all")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--reranker-model", default=FINAL_RERANKER_MODEL)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--output", default="results/context_metrics.csv")
    args = parser.parse_args()

    documents = load_documents(args.documents)
    questions = pd.read_parquet(args.questions).head(args.limit)
    rows = [row for _, row in questions.iterrows()]
    retriever = HybridRetriever(
        documents,
        alpha=args.alpha,
        embeddings_path=args.embeddings,
    )
    reranker = None if args.no_reranker else CrossEncoderReranker(args.reranker_model)
    pipeline = LegalQAPipeline(retriever, reranker=reranker)
    result_lists = [
        pipeline.retriever.search(row["question"], top_k=20, category=args.category)
        for row in rows
    ]
    if reranker:
        result_lists = [
            reranker.rerank(row["question"], results, top_k=5)
            for row, results in zip(rows, result_lists)
        ]
    metrics = evaluate_context_results(rows, result_lists)
    print(metrics)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "questions": len(rows),
        "category": args.category,
        "reranker": not args.no_reranker,
        **metrics,
    }]).to_csv(output_path, index=False)
    print(f"บันทึกผลแล้ว: {output_path}")


if __name__ == "__main__":
    main()