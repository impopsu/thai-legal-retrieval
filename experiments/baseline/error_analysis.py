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

tokenized_documents = [
    word_tokenize(text, engine="newmm")
    for text in document_texts
]

bm25 = BM25Okapi(tokenized_documents)

document_embeddings = model.encode(
    document_texts,
    convert_to_tensor=True,
    show_progress_bar=True
)

found_cases = []

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

    hybrid_scores = (
        0.5 * bm25_norm
        + 0.5 * semantic_norm
    )

    bm25_top5 = np.argsort(
        bm25_scores
    )[::-1][:5]

    hybrid_top5 = np.argsort(
        hybrid_scores
    )[::-1][:5]

    bm25_found = any(
        documents.iloc[index]["unique_key"] in positive_keys
        for index in bm25_top5
    )

    hybrid_found = any(
        documents.iloc[index]["unique_key"] in positive_keys
        for index in hybrid_top5
    )

    if not bm25_found and hybrid_found:

        correct_key = next(
            key for key in positive_keys
            if key in set(
                documents.iloc[index]["unique_key"]
                for index in hybrid_top5
            )
        )

        correct_index = documents.index[
            documents["unique_key"] == correct_key
        ][0]

        found_cases.append({
            "question": query,
            "correct_law": documents.iloc[correct_index]["law_title"],
            "correct_section": documents.iloc[correct_index]["section"],
            "correct_context": documents.iloc[correct_index]["context"],
            "bm25_top": [
                (
                    documents.iloc[index]["law_title"],
                    documents.iloc[index]["section"]
                )
                for index in bm25_top5
            ],
            "hybrid_top": [
                (
                    documents.iloc[index]["law_title"],
                    documents.iloc[index]["section"]
                )
                for index in hybrid_top5
            ]
        })

    if len(found_cases) >= 10:
        break

print("\n===== BM25 พลาด แต่ Hybrid หาเจอ =====")

for i, case in enumerate(found_cases, start=1):

    print(f"\n{'=' * 70}")
    print(f"ตัวอย่างที่ {i}")

    print("\nคำถาม:")
    print(case["question"])

    print("\nคำตอบที่ถูก:")
    print(case["correct_law"])
    print(f"มาตรา {case['correct_section']}")

    print("\nBM25 Top 5:")
    for rank, item in enumerate(case["bm25_top"], start=1):
        print(f"{rank}. {item[0]} มาตรา {item[1]}")

    print("\nHybrid Top 5:")
    for rank, item in enumerate(case["hybrid_top"], start=1):
        print(f"{rank}. {item[0]} มาตรา {item[1]}")