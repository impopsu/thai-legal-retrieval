import pandas as pd
from rank_bm25 import BM25Okapi
from pythainlp.tokenize import word_tokenize

documents = pd.read_csv(
    "data/processed/legal_documents.csv",
    encoding="utf-8-sig"
)

tokenized_documents = [
    word_tokenize(text, engine="newmm")
    for text in documents["context"].fillna("")
]

bm25 = BM25Okapi(tokenized_documents)

query = input("พิมพ์คำถามกฎหมาย: ")

tokenized_query = word_tokenize(
    query,
    engine="newmm"
)

print("\nคำที่ระบบตัดได้:")
print(tokenized_query)

scores = bm25.get_scores(tokenized_query)

top_k = 5
top_indices = scores.argsort()[::-1][:top_k]

print("\n===== ผลการค้นหา =====")

for rank, index in enumerate(top_indices, start=1):
    row = documents.iloc[index]

    print(f"\nอันดับ {rank}")
    print(f"กฎหมาย: {row['law_title']}")
    print(f"มาตรา: {row['section']}")
    print(f"คะแนน: {scores[index]:.4f}")
    print(f"เนื้อหา: {row['context'][:300]}...")
