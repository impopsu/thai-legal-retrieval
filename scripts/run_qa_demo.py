import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_qa import BM25Retriever, HybridRetriever, LegalQAPipeline, load_documents
from legal_qa.reranking import CrossEncoderReranker
from legal_qa.qa import TransformersGenerator


FINAL_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Thai legal QA demo")
    parser.add_argument(
        "--documents",
        default="data/processed/legal_documents.csv",
    )
    parser.add_argument(
        "--retriever",
        choices=("bm25", "hybrid"),
        default="hybrid",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--embeddings",
        default="results/minilm_document_embeddings.pt",
    )
    parser.add_argument(
        "--generator-model",
        help="Optional Hugging Face text2text-generation model",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--category", default="all")
    parser.add_argument(
        "--reranker-model",
        default=FINAL_RERANKER_MODEL,
        help="Hugging Face cross-encoder reranker model",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Disable reranking for a faster baseline demo",
    )
    return parser.parse_args()


def print_response(response) -> None:
    print("\n===== แหล่งข้อมูลที่ค้นพบ =====")
    for rank, result in enumerate(response.sources, start=1):
        document = result.document
        print(f"\n{rank}. {document.law_title} มาตรา {document.section}")
        print(f"คะแนน: {result.score:.4f}")
        print(document.context[:500])

    if response.answer:
        print("\n===== คำตอบ =====")
        print(response.answer)
    else:
        print("\nยังไม่ได้โหลด answer generator")
        print("ระบบแสดง evidence และ grounded prompt เพื่อใช้ต่อกับ LLM")


def main() -> None:
    args = parse_args()
    documents = load_documents(args.documents)

    if args.retriever == "bm25":
        retriever = BM25Retriever(documents)
    else:
        retriever = HybridRetriever(
            documents,
            alpha=args.alpha,
            embeddings_path=args.embeddings,
        )

    generator = None
    if args.generator_model:
        generator = TransformersGenerator(args.generator_model)

    reranker = None
    if not args.no_reranker:
        reranker = CrossEncoderReranker(args.reranker_model)

    pipeline = LegalQAPipeline(
        retriever,
        reranker=reranker,
        generator=generator,
    )
    print("พิมพ์ exit เพื่อออกจากโปรแกรม")

    while True:
        question = input("\nคำถามกฎหมาย: ").strip()
        if question.lower() == "exit":
            break
        if question:
            print_response(
                pipeline.answer(
                    question,
                    top_k=args.top_k,
                    category=args.category,
                )
            )


if __name__ == "__main__":
    main()