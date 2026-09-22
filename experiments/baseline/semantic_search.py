import pandas as pd
from sentence_transformers import SentenceTransformer, util

documents = pd.read_csv(
    "data/processed/legal_documents.csv",
    encoding="utf-8-sig"
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

query = input("\nพิมพ์คำถามกฎหมาย: ")

query_embedding = model.encode(
    query,
    convert_to_tensor=True
)

scores = util.cos_sim(
    query_embedding,
    document_embeddings
)[0]

top_k = 5
top_indices = scores.argsort(descending=True)[:top_k]

print("\n===== ผลการค้นหา Semantic Search =====")

for rank, index in enumerate(top_indices, start=1):
    index = index.item()
    row = documents.iloc[index]

    print(f"\nอันดับ {rank}")
    print(f"กฎหมาย: {row['law_title']}")
    print(f"มาตรา: {row['section']}")
    print(f"คะแนน: {scores[index].item():.4f}")
    print(f"เนื้อหา: {row['context'][:300]}...")