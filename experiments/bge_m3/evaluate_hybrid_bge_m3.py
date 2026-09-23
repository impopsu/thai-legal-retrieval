import os
import pandas as pd
import numpy as np
import torch

from rank_bm25 import BM25Okapi
from pythainlp.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


print("===== Hybrid BM25 + BGE-M3 Evaluation =====")


# =========================
# Paths
# =========================

DOCUMENT_EMBEDDINGS_PATH = (
    "results/bge_m3_document_embeddings.pt"
)

QUERY_EMBEDDINGS_PATH = (
    "results/bge_m3_query_embeddings.pt"
)

RESULTS_PATH = (
    "results/hybrid_bge_m3_results.csv"
)

EMBEDDING_BATCH_SIZE = int(
    os.getenv("BGE_BATCH_SIZE", "8")
)

MAX_TEST_QUERIES = int(
    os.getenv("BGE_MAX_QUERIES", "0")
)


# =========================
# Load data
# =========================

documents = pd.read_csv(
    "data/processed/legal_documents.csv",
    encoding="utf-8-sig"
)

test = pd.read_parquet(
    "data/raw/test-00000-of-00001.parquet"
)

if MAX_TEST_QUERIES > 0:
    test = test.head(MAX_TEST_QUERIES).copy()

document_texts = documents["context"].fillna("").tolist()
questions = test["question"].fillna("").tolist()

print(f"จำนวน Documents: {len(document_texts)}")
print(f"จำนวน Test: {len(test)}")


# =========================
# BM25
# =========================

print("\nกำลังสร้าง BM25...")

tokenized_documents = [
    word_tokenize(text, engine="newmm")
    for text in document_texts
]

bm25 = BM25Okapi(tokenized_documents)

print("สร้าง BM25 สำเร็จ")


# =========================
# BGE-M3
# =========================

print("\nกำลังโหลด BGE-M3...")

model = SentenceTransformer("BAAI/bge-m3")

print("โหลด BGE-M3 สำเร็จ")


# =========================
# Document embeddings
# =========================

if os.path.exists(DOCUMENT_EMBEDDINGS_PATH):

    print("\nพบ Document embeddings แล้ว")
    print("กำลังโหลดจากไฟล์...")

    document_embeddings = torch.load(
        DOCUMENT_EMBEDDINGS_PATH,
        weights_only=True
    )

    print(
        f"โหลด Document embeddings สำเร็จ "
        f"shape={document_embeddings.shape}"
    )

else:

    print("\nไม่พบ Document embeddings")
    print("กำลังสร้าง BGE-M3 embeddings ของ Documents...")

    document_embeddings = model.encode(
        document_texts,
        batch_size=4,
        convert_to_tensor=True,
        show_progress_bar=True
    )

    os.makedirs("results", exist_ok=True)

    torch.save(
        document_embeddings.cpu(),
        DOCUMENT_EMBEDDINGS_PATH
    )

    print(
        f"\nบันทึก Document embeddings แล้ว: "
        f"{DOCUMENT_EMBEDDINGS_PATH}"
    )


document_embeddings = document_embeddings.to(
    model.device
)


# =========================
# Query embeddings
# =========================

cached_query_embeddings = None

if os.path.exists(QUERY_EMBEDDINGS_PATH):

    cached_query_embeddings = torch.load(
        QUERY_EMBEDDINGS_PATH,
        weights_only=True
    )

    if cached_query_embeddings.ndim != 2:
        cached_query_embeddings = None
    elif cached_query_embeddings.shape[1] != model.get_embedding_dimension():
        cached_query_embeddings = None

if (
    cached_query_embeddings is not None
    and cached_query_embeddings.shape[0] >= len(questions)
):

    print("\nพบ Query embeddings แล้ว")
    print("กำลังโหลดจากไฟล์...")

    query_embeddings = cached_query_embeddings[:len(questions)]

    print(
        f"โหลด Query embeddings สำเร็จ "
        f"shape={query_embeddings.shape}"
    )

else:

    if cached_query_embeddings is not None:
        query_embeddings = cached_query_embeddings
        start_index = len(query_embeddings)
        print(
            f"\nพบ Query embeddings บางส่วน "
            f"({start_index}/{len(questions)})"
        )
    else:
        query_embeddings = torch.empty(
            (0, model.get_embedding_dimension())
        )
        start_index = 0
        print("\nไม่พบ Query embeddings")

    print("กำลังสร้าง BGE-M3 embeddings ของ Questions...")

    os.makedirs("results", exist_ok=True)

    for batch_start in range(
        start_index,
        len(questions),
        EMBEDDING_BATCH_SIZE
    ):

        batch_end = min(
            batch_start + EMBEDDING_BATCH_SIZE,
            len(questions)
        )

        batch_embeddings = model.encode(
            questions[batch_start:batch_end],
            batch_size=EMBEDDING_BATCH_SIZE,
            convert_to_tensor=True,
            show_progress_bar=True
        ).cpu()

        query_embeddings = torch.cat(
            [query_embeddings, batch_embeddings]
        )

        torch.save(
            query_embeddings,
            QUERY_EMBEDDINGS_PATH
        )

        print(
            f"บันทึก Query embeddings แล้ว "
            f"{batch_end}/{len(questions)}"
        )

    print(
        f"\nบันทึก Query embeddings สำเร็จ: "
        f"{QUERY_EMBEDDINGS_PATH}"
    )


query_embeddings = query_embeddings.to(
    model.device
)


# =========================
# Normalization
# =========================

def min_max_normalize(scores):

    scores = np.asarray(scores)

    min_score = scores.min()
    max_score = scores.max()

    if max_score == min_score:
        return np.zeros_like(scores)

    return (
        (scores - min_score)
        / (max_score - min_score)
    )


# =========================
# Evaluation
# =========================

alphas = [0.25, 0.5, 0.75]

metrics = {
    alpha: {
        "recall_1": 0,
        "recall_3": 0,
        "recall_5": 0,
        "reciprocal_rank_sum": 0
    }
    for alpha in alphas
}

print("\nกำลังคำนวณ semantic scores ของทุกคำถาม...")

with torch.inference_mode():
    normalized_documents = torch.nn.functional.normalize(
        document_embeddings,
        p=2,
        dim=1
    )
    normalized_queries = torch.nn.functional.normalize(
        query_embeddings,
        p=2,
        dim=1
    )
    semantic_scores_matrix = torch.mm(
        normalized_queries,
        normalized_documents.T
    ).cpu().numpy()

print("คำนวณ semantic scores สำเร็จ")

progress = tqdm(
    test.iterrows(),
    total=len(test),
    desc="Evaluating all alphas",
    unit="query"
)

for i, row in progress:

    question = row["question"]

    positive_contexts = row["positive_contexts"]

    positive_keys = {
        context["unique_key"]
        for context in positive_contexts
    }

    # Calculate each score once per query, then reuse it for all alphas.
    tokenized_query = word_tokenize(
        question,
        engine="newmm"
    )

    bm25_scores = min_max_normalize(
        bm25.get_scores(tokenized_query)
    )

    semantic_scores = min_max_normalize(
        semantic_scores_matrix[i]
    )

    for alpha in alphas:

        hybrid_scores = (
            alpha * semantic_scores
            + (1 - alpha) * bm25_scores
        )

        ranked_indices = np.argsort(
            hybrid_scores
        )[::-1]

        ranked_keys = [
            documents.iloc[index]["unique_key"]
            for index in ranked_indices
        ]

        alpha_metrics = metrics[alpha]

        if any(key in positive_keys for key in ranked_keys[:1]):
            alpha_metrics["recall_1"] += 1

        if any(key in positive_keys for key in ranked_keys[:3]):
            alpha_metrics["recall_3"] += 1

        if any(key in positive_keys for key in ranked_keys[:5]):
            alpha_metrics["recall_5"] += 1

        for rank, key in enumerate(
            ranked_keys,
            start=1
        ):

            if key in positive_keys:

                alpha_metrics["reciprocal_rank_sum"] += 1 / rank
                break

n = len(test)
results = []

for alpha in alphas:

    alpha_metrics = metrics[alpha]
    recall_1 = alpha_metrics["recall_1"] / n
    recall_3 = alpha_metrics["recall_3"] / n
    recall_5 = alpha_metrics["recall_5"] / n
    mrr = alpha_metrics["reciprocal_rank_sum"] / n

    print(f"\n===== Alpha = {alpha} =====")
    print(f"Recall@1: {recall_1:.4f}")
    print(f"Recall@3: {recall_3:.4f}")
    print(f"Recall@5: {recall_5:.4f}")
    print(f"MRR: {mrr:.4f}")

    results.append({
        "alpha": alpha,
        "Recall@1": recall_1,
        "Recall@3": recall_3,
        "Recall@5": recall_5,
        "MRR": mrr
    })

results_df = pd.DataFrame(results)

os.makedirs("results", exist_ok=True)

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

print(f"\nบันทึกผลแล้ว: {RESULTS_PATH}")


# =========================
# Final summary
# =========================

results_df = pd.DataFrame(results)

print("\n===== Summary =====")

print(
    results_df.to_string(index=False)
)

print(
    f"\nบันทึกผลแล้ว: {RESULTS_PATH}"
)