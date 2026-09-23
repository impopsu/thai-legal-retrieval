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
    "BAAI/bge-m3"
)

document_texts = documents["context"].fillna("").tolist()

print("กำลังสร้าง BGE-M3 embeddings ของเอกสาร...")

document_embeddings = model.encode(
    document_texts,
    batch_size=4,
    convert_to_tensor=True,
    show_progress_bar=True
)

recall_1 = 0
recall_3 = 0
recall_5 = 0
mrr = 0

print("กำลังประเมิน BGE-M3...")

for i, row in test.iterrows():

    query = row["question"]

    positive_contexts = row["positive_contexts"]

    positive_keys = {
        context["unique_key"]
        for context in positive_contexts
    }

    query_embedding = model.encode(
        query,
        convert_to_tensor=True
    )

    scores = util.cos_sim(
        query_embedding,
        document_embeddings
    )[0]

    top_k = min(5, len(documents))

    top_results = scores.topk(top_k).indices.tolist()

    retrieved_keys = [
        documents.iloc[idx]["unique_key"]
        for idx in top_results
    ]

    if retrieved_keys[0] in positive_keys:
        recall_1 += 1

    if any(key in positive_keys for key in retrieved_keys[:3]):
        recall_3 += 1

    if any(key in positive_keys for key in retrieved_keys[:5]):
        recall_5 += 1

    rank = None

    for r, key in enumerate(retrieved_keys, start=1):
        if key in positive_keys:
            rank = r
            break

    if rank is not None:
        mrr += 1 / rank

    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1}/{len(test)}")

n = len(test)

print()
print("===== BGE-M3 Evaluation =====")
print(f"จำนวน Test: {n}")
print(f"Recall@1: {recall_1 / n:.4f}")
print(f"Recall@3: {recall_3 / n:.4f}")
print(f"Recall@5: {recall_5 / n:.4f}")
print(f"MRR: {mrr / n:.4f}")