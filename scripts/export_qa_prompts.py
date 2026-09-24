import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_qa import HybridRetriever, LegalQAPipeline, load_documents
from legal_qa.reranking import CrossEncoderReranker


FINAL_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export grounded prompts without generating answers"
    )
    parser.add_argument("--documents", default="data/processed/legal_documents.csv")
    parser.add_argument(
        "--questions",
        "--validation",
        dest="questions",
        default="data/processed/validation_retrieval.parquet",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--embeddings",
        default="results/minilm_document_embeddings.pt",
    )
    parser.add_argument("--reranker-model", default=FINAL_RERANKER_MODEL)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument(
        "--output",
        default="results/qa_prompts_validation.jsonl",
    )
    args = parser.parse_args()

    documents = load_documents(args.documents)
    validation = pd.read_parquet(args.questions).head(args.limit)
    retriever = HybridRetriever(
        documents,
        alpha=args.alpha,
        embeddings_path=args.embeddings,
    )
    reranker = None
    if not args.no_reranker:
        reranker = CrossEncoderReranker(args.reranker_model)
    pipeline = LegalQAPipeline(retriever, reranker=reranker)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for _, row in validation.iterrows():
            response = pipeline.answer(row["question"], top_k=5)
            json.dump(
                {
                    "question": response.question,
                    "prompt": response.prompt,
                    "sources": [
                        {
                            "unique_key": result.document.unique_key,
                            "law_title": result.document.law_title,
                            "section": result.document.section,
                            "score": result.score,
                        }
                        for result in response.sources
                    ],
                },
                output,
                ensure_ascii=False,
            )
            output.write("\n")

    print(f"ส่งออก {len(validation)} prompts แล้ว: {output_path}")


if __name__ == "__main__":
    main()