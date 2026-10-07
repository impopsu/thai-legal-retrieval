import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_qa import BM25Retriever, HybridRetriever, SemanticRetriever, load_documents
from legal_qa.reranking import CrossEncoderReranker


FINAL_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def precision_recall_f1(ranking, positive_set, k):
    retrieved = set(ranking[:k])
    tp = len(retrieved & positive_set)
    fp = len(retrieved - positive_set)
    fn = len(positive_set - retrieved)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def mrr(ranking, positive_set):
    for rank, index in enumerate(ranking[:5], start=1):
        if index in positive_set:
            return 1.0 / rank
    return 0.0


def evaluate_method(rows, ranking_fn, ks=(1, 3, 5)):
    score_columns = {k: {"precision": 0.0, "recall": 0.0, "f1": 0.0} for k in ks}
    reciprocal_rank_sum = 0.0
    valid_questions = 0

    for row in rows:
        ranking = ranking_fn(row)
        positive_set = set(_positive_indices_for_row(row))
        if not positive_set:
            continue
        valid_questions += 1
        for k in ks:
            precision, recall, f1 = precision_recall_f1(ranking, positive_set, k)
            score_columns[k]["precision"] += precision
            score_columns[k]["recall"] += recall
            score_columns[k]["f1"] += f1
        reciprocal_rank_sum += mrr(ranking, positive_set)

    if valid_questions == 0:
        raise ValueError("no valid questions for retrieval F1 evaluation")

    total_questions = len(rows)
    skipped_no_ground_truth = sum(1 for row in rows if not _positive_indices_for_row(row))
    skipped_missing_document = 0
    metrics = {
        "model": "",
        "total_questions": total_questions,
        "valid_questions": valid_questions,
        "skipped_missing_document": skipped_missing_document,
        "skipped_no_ground_truth": skipped_no_ground_truth,
    }
    for k in ks:
        metrics[f"precision_at_{k}"] = score_columns[k]["precision"] / valid_questions
        metrics[f"recall_at_{k}"] = score_columns[k]["recall"] / valid_questions
        metrics[f"f1_at_{k}"] = score_columns[k]["f1"] / valid_questions
    metrics["mrr_at_5"] = reciprocal_rank_sum / valid_questions
    return metrics


def _positive_indices_for_row(row):
    positive_contexts = row.get("positive_contexts")
    if positive_contexts is None:
        return []
    positive_contexts = list(positive_contexts)
    return [str(item["unique_key"]) for item in positive_contexts]


def summarize_rankings(rows, rankings, document_ids, ks=(1, 3, 5)):
    if len(rankings) != len(rows):
        raise ValueError("rankings and questions must have the same length")
    positive_all = [_positive_indices_for_row(row) for row in rows]

    valid_questions = 0
    total_questions = len(rows)
    skipped_no_ground_truth = 0
    skipped_missing_document = 0

    valid_positive_indices = []
    for positives in positive_all:
        valid = [doc_id for doc_id in positives if doc_id in document_ids]
        valid_positive_indices.append(valid)
        if not positives:
            skipped_no_ground_truth += 1
            continue
        if not valid:
            skipped_missing_document += 1
            continue
        valid_questions += 1

    metrics = {"total_questions": total_questions, "valid_questions": valid_questions, "skipped_missing_document": skipped_missing_document, "skipped_no_ground_truth": skipped_no_ground_truth}
    for k in ks:
        precision_total = 0.0
        recall_total = 0.0
        f1_total = 0.0
        for ranking, positive_set in zip(rankings, valid_positive_indices):
            if not positive_set:
                continue
            precision, recall, f1 = precision_recall_f1(ranking, set(positive_set), k)
            precision_total += precision
            recall_total += recall
            f1_total += f1
        metrics[f"precision_at_{k}"] = precision_total / valid_questions if valid_questions else 0.0
        metrics[f"recall_at_{k}"] = recall_total / valid_questions if valid_questions else 0.0
        metrics[f"f1_at_{k}"] = f1_total / valid_questions if valid_questions else 0.0

    reciprocal_rank_total = 0.0
    for ranking, positive_set in zip(rankings, valid_positive_indices):
        if not positive_set:
            continue
        reciprocal_rank_total += mrr(ranking, set(positive_set))
    metrics["mrr_at_5"] = reciprocal_rank_total / valid_questions if valid_questions else 0.0
    return metrics


def build_rankings(rows, retriever, reranker=None, candidate_k=20, top_k=5):
    rankings = []
    for row in rows:
        question = row["question"]
        if reranker is None:
            results = retriever.search(question, top_k=top_k)
        else:
            candidates = retriever.search(question, top_k=candidate_k)
            results = reranker.rerank(question, candidates, top_k)
        rankings.append([result.document.unique_key for result in results])
    return rankings


def main():
    parser = argparse.ArgumentParser(
        description="Compute precision, recall and F1 at k for legal retrieval"
    )
    parser.add_argument("--questions", default="data/raw/test-00000-of-00001.parquet")
    parser.add_argument("--documents", default="data/processed/legal_documents_categorized.csv")
    parser.add_argument("--output", default="results/retrieval_f1_test.csv")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("bm25", "minilm", "hybrid", "hybrid_reranker", "bge_m3", "hybrid_bge"),
        default=["bm25", "minilm", "hybrid", "hybrid_reranker"],
    )
    args = parser.parse_args()

    documents = load_documents(args.documents)
    questions = pd.read_parquet(args.questions).head(args.limit)
    rows = [row for _, row in questions.iterrows()]
    if not rows:
        raise ValueError("No questions available for retrieval F1 evaluation")
    output_rows = []

    retrievers = {}
    if "bm25" in args.methods:
        retrievers["BM25"] = (BM25Retriever(documents), None)
    if "minilm" in args.methods:
        retrievers["Semantic MiniLM"] = (
            SemanticRetriever(
                documents,
                embeddings_path="results/minilm_document_embeddings.pt",
            ),
            None,
        )
    if "hybrid" in args.methods or "hybrid_reranker" in args.methods:
        hybrid_minilm = HybridRetriever(
            documents,
            alpha=args.alpha,
            embeddings_path="results/minilm_document_embeddings.pt",
        )
        retrievers["Hybrid MiniLM"] = (hybrid_minilm, None)
        if "hybrid_reranker" in args.methods:
            retrievers["Hybrid MiniLM + Cross-Encoder Reranker"] = (
                hybrid_minilm,
                CrossEncoderReranker(FINAL_RERANKER_MODEL),
            )
    if "bge_m3" in args.methods:
        retrievers["BGE-M3"] = (
            SemanticRetriever(
                documents,
                model_name="BAAI/bge-m3",
                embeddings_path="results/bge_m3_document_embeddings.pt",
                batch_size=4,
            ),
            None,
        )
    if "hybrid_bge" in args.methods:
        retrievers["Hybrid BGE-M3"] = (
            HybridRetriever(
                documents,
                model_name="BAAI/bge-m3",
                embeddings_path="results/bge_m3_document_embeddings.pt",
                alpha=args.alpha,
                batch_size=4,
            ),
            None,
        )

    document_ids = {doc.unique_key for doc in documents}
    for name, (retriever, reranker) in retrievers.items():
        rankings = build_rankings(
            rows,
            retriever,
            reranker=reranker,
            candidate_k=args.candidate_k,
            top_k=args.top_k,
        )
        metrics = summarize_rankings(rows, rankings, document_ids)
        metrics["model"] = name
        metrics["split"] = Path(args.questions).stem
        metrics["k"] = args.top_k
        output_rows.append(metrics)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(out, index=False)
    print(pd.DataFrame(output_rows).to_string(index=False))
    print(f"Saved F1 metrics to {out}")


if __name__ == "__main__":
    main()
