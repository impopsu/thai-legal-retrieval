import pandas as pd
from rank_bm25 import BM25Okapi
from pythainlp.tokenize import word_tokenize

documents = pd.read_csv(
    "data/processed/legal_documents.csv",
    encoding="utf-8-sig"
)

train = pd.read_parquet(
    "data/raw/train-00000-of-00001.parquet"
)

test = pd.read_parquet(
    "data/raw/test-00000-of-00001.parquet"
)

tokenized_documents = [
    word_tokenize(text, engine="newmm")
    for text in documents["context"].fillna("")
]

bm25 = BM25Okapi(tokenized_documents)

document_indices = {
    key: index
    for index, key in enumerate(documents["unique_key"])
}

recall_1 = 0
recall_3 = 0
recall_5 = 0
mrr = 0

total = len(test)

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

    scores = bm25.get_scores(tokenized_query)

    top_indices = scores.argsort()[::-1][:5]

    retrieved_keys = [
        documents.iloc[index]["unique_key"]
        for index in top_indices
    ]

    if retrieved_keys[0] in positive_keys:
        recall_1 += 1

    if any(key in positive_keys for key in retrieved_keys[:3]):
        recall_3 += 1

    if any(key in positive_keys for key in retrieved_keys[:5]):
        recall_5 += 1

    rank = None

    for position, key in enumerate(retrieved_keys, start=1):
        if key in positive_keys:
            rank = position
            break

    if rank is not None:
        mrr += 1 / rank

    if (i + 1) % 500 == 0:
        print(f"ประมวลผลแล้ว {i + 1}/{total}")

print("\n===== BM25 Evaluation =====")

print(f"จำนวน Test: {total}")

print(f"Recall@1: {recall_1 / total:.4f}")
print(f"Recall@3: {recall_3 / total:.4f}")
print(f"Recall@5: {recall_5 / total:.4f}")

print(f"MRR: {mrr / total:.4f}")