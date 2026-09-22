import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from pythainlp.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv(
    "data/processed/legal_documents.csv"
)

documents = df["context"].fillna("").tolist()


def tokenize(text):
    return word_tokenize(
        str(text),
        engine="newmm"
    )


tokenized_documents = [
    tokenize(doc)
    for doc in documents
]

bm25 = BM25Okapi(tokenized_documents)

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

document_embeddings = model.encode(
    documents,
    normalize_embeddings=True,
    show_progress_bar=True
)


def min_max_normalize(scores):
    min_score = np.min(scores)
    max_score = np.max(scores)

    if max_score == min_score:
        return np.zeros_like(scores)

    return (scores - min_score) / (max_score - min_score)


def hybrid_search(query, top_k=5, alpha=0.5):

    query_tokens = tokenize(query)

    bm25_scores = np.array(
        bm25.get_scores(query_tokens)
    )

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    semantic_scores = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]

    bm25_norm = min_max_normalize(bm25_scores)
    semantic_norm = min_max_normalize(semantic_scores)

    hybrid_scores = (
        alpha * bm25_norm +
        (1 - alpha) * semantic_norm
    )

    top_indices = np.argsort(
        hybrid_scores
    )[::-1][:top_k]

    return top_indices, hybrid_scores


print("=" * 60)
print("ระบบค้นหากฎหมายไทยด้วย Hybrid Search")
print("=" * 60)
print("พิมพ์ exit เพื่อออกจากโปรแกรม")
print()

while True:

    query = input("คำถามกฎหมาย: ").strip()

    if query.lower() == "exit":
        print("จบการทำงาน")
        break

    if not query:
        continue

    top_indices, scores = hybrid_search(query)

    print()
    print("ผลการค้นหา")
    print("-" * 60)

    for rank, idx in enumerate(top_indices, start=1):

        row = df.iloc[idx]

        print(f"\nอันดับ {rank}")
        print(f"กฎหมาย: {row['law_title']}")
        print(f"มาตรา: {row['section']}")
        print(f"คะแนน: {scores[idx]:.4f}")
        print(f"เนื้อหา: {row['context']}")

    print()
    print("=" * 60)