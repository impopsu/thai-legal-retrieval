import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from pythainlp.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util

documents = pd.read_csv(
    "data/processed/legal_documents.csv",
    encoding="utf-8-sig"
)

test = pd.read_parquet(
    "data/raw/test-00000-of-00001.parquet"
)

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

document_texts = documents["context"].fillna("").tolist()

print("กำลังสร้าง BM25...")

tokenized_documents = [
    word_tokenize(text, engine="newmm")
    for text in document_texts
]

bm25 = BM25Okapi(tokenized_documents)

print("กำลังสร้าง Semantic Embeddings...")

document_embeddings = model.encode(
    document_texts,
    convert_to_tensor=True,
    show_progress_bar=True
)

alphas = [0.25, 0.5, 0.75]

results = {}

for alpha in alphas:
    results[alpha] = {
        "recall_1": 0,
        "recall_3": 0,
        "recall_5": 0,
        "mrr": 0
    }

total = len(test)

print("\nกำลังประเมิน Hybrid Search...")

for i, row in test.iterrows():

    query = row["question"]

    positive_keys = {
        item["unique_key"]
        for item in row["positive_contexts"]
    }

    tokenized_query = word_tokenize(
        query,
        engine="newmm"
    )

    bm25_scores = np.array(
        bm25.get_scores(tokenized_query)
    )

    query_embedding = model.encode(
        query,
        convert_to_tensor=True
    )

    semantic_scores = util.cos_sim(
        query_embedding,
        document_embeddings
    )[0].cpu().numpy()

    bm25_min = bm25_scores.min()
    bm25_max = bm25_scores.max()

    if bm25_max > bm25_min:
        bm25_norm = (
            (bm25_scores - bm25_min)
            / (bm25_max - bm25_min)
        )
    else:
        bm25_norm = np.zeros_like(bm25_scores)

    semantic_min = semantic_scores.min()
    semantic_max = semantic_scores.max()

    if semantic_max > semantic_min:
        semantic_norm = (
            (semantic_scores - semantic_min)
            / (semantic_max - semantic_min)
        )
    else:
        semantic_norm = np.zeros_like(semantic_scores)

    for alpha in alphas:

        hybrid_scores = (
            alpha * bm25_norm
            + (1 - alpha) * semantic_norm
        )

        top_indices = np.argsort(
            hybrid_scores
        )[::-1][:5]

        retrieved_keys = [
            documents.iloc[index]["unique_key"]
            for index in top_indices
        ]

        if retrieved_keys[0] in positive_keys:
            results[alpha]["recall_1"] += 1

        if any(
            key in positive_keys
            for key in retrieved_keys[:3]
        ):
            results[alpha]["recall_3"] += 1

        if any(
            key in positive_keys
            for key in retrieved_keys[:5]
        ):
            results[alpha]["recall_5"] += 1

        rank = None

        for position, key in enumerate(
            retrieved_keys,
            start=1
        ):
            if key in positive_keys:
                rank = position
                break

        if rank is not None:
            results[alpha]["mrr"] += 1 / rank

    if (i + 1) % 500 == 0:
        print(f"ประมวลผลแล้ว {i + 1}/{total}")

print("\n===== Hybrid Search Evaluation =====")

for alpha in alphas:

    result = results[alpha]

    print(f"\nAlpha = {alpha}")

    print(
        f"Recall@1: "
        f"{result['recall_1'] / total:.4f}"
    )

    print(
        f"Recall@3: "
        f"{result['recall_3'] / total:.4f}"
    )

    print(
        f"Recall@5: "
        f"{result['recall_5'] / total:.4f}"
    )

    print(
        f"MRR: "
        f"{result['mrr'] / total:.4f}"
    )