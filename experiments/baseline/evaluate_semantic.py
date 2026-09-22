import pandas as pd
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

print("กำลังสร้าง embeddings ของเอกสาร...")

document_embeddings = model.encode(
    document_texts,
    convert_to_tensor=True,
    show_progress_bar=True
)

recall_1 = 0
recall_3 = 0
recall_5 = 0
mrr = 0

total = len(test)

print("\nกำลังประเมิน Test...")

for i, row in test.iterrows():
    query = row["question"]

    positive_keys = {
        item["unique_key"]
        for item in row["positive_contexts"]
    }

    query_embedding = model.encode(
        query,
        convert_to_tensor=True
    )

    scores = util.cos_sim(
        query_embedding,
        document_embeddings
    )[0]

    top_indices = scores.argsort(descending=True)[:5]

    retrieved_keys = [
        documents.iloc[index.item()]["unique_key"]
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

print("\n===== Semantic Search Evaluation =====")

print(f"จำนวน Test: {total}")

print(f"Recall@1: {recall_1 / total:.4f}")
print(f"Recall@3: {recall_3 / total:.4f}")
print(f"Recall@5: {recall_5 / total:.4f}")

print(f"MRR: {mrr / total:.4f}")