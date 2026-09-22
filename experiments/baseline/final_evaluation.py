import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pythainlp.tokenize import word_tokenize

documents = pd.read_csv(
    "data/processed/legal_documents.csv"
)

test = pd.read_parquet(
    "data/raw/test-00000-of-00001.parquet"
)

doc_texts = documents["context"].fillna("").tolist()

print("Loading BM25...")

tokenized_docs = [
    word_tokenize(text, engine="newmm")
    for text in doc_texts
]

bm25 = BM25Okapi(tokenized_docs)

print("Loading semantic model...")

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("Encoding documents...")

doc_embeddings = model.encode(
    doc_texts,
    batch_size=32,
    show_progress_bar=True
)


def normalize(scores):
    min_score = scores.min()
    max_score = scores.max()

    if max_score == min_score:
        return np.zeros_like(scores)

    return (scores - min_score) / (max_score - min_score)


def evaluate_bm25():

    recall1 = 0
    recall3 = 0
    recall5 = 0
    reciprocal_rank = []

    for _, row in test.iterrows():

        positive_keys = {
            x["unique_key"]
            for x in row["positive_contexts"]
        }

        query_tokens = word_tokenize(
            row["question"],
            engine="newmm"
        )

        scores = bm25.get_scores(query_tokens)
        ranking = np.argsort(scores)[::-1]

        top1 = ranking[:1]
        top3 = ranking[:3]
        top5 = ranking[:5]

        keys1 = {
            documents.iloc[i]["unique_key"]
            for i in top1
        }

        keys3 = {
            documents.iloc[i]["unique_key"]
            for i in top3
        }

        keys5 = {
            documents.iloc[i]["unique_key"]
            for i in top5
        }

        if positive_keys & keys1:
            recall1 += 1

        if positive_keys & keys3:
            recall3 += 1

        if positive_keys & keys5:
            recall5 += 1

        rank = None

        for r, i in enumerate(ranking[:5], start=1):

            if documents.iloc[i]["unique_key"] in positive_keys:
                rank = r
                break

        reciprocal_rank.append(
            1 / rank if rank else 0
        )

    n = len(test)

    return {
        "Recall@1": recall1 / n,
        "Recall@3": recall3 / n,
        "Recall@5": recall5 / n,
        "MRR": np.mean(reciprocal_rank)
    }


def evaluate_semantic():

    recall1 = 0
    recall3 = 0
    recall5 = 0
    reciprocal_rank = []

    for _, row in test.iterrows():

        positive_keys = {
            x["unique_key"]
            for x in row["positive_contexts"]
        }

        query_embedding = model.encode(
            [row["question"]],
            show_progress_bar=False
        )[0]

        scores = cosine_similarity(
            [query_embedding],
            doc_embeddings
        )[0]

        ranking = np.argsort(scores)[::-1]

        top1 = ranking[:1]
        top3 = ranking[:3]
        top5 = ranking[:5]

        keys1 = {
            documents.iloc[i]["unique_key"]
            for i in top1
        }

        keys3 = {
            documents.iloc[i]["unique_key"]
            for i in top3
        }

        keys5 = {
            documents.iloc[i]["unique_key"]
            for i in top5
        }

        if positive_keys & keys1:
            recall1 += 1

        if positive_keys & keys3:
            recall3 += 1

        if positive_keys & keys5:
            recall5 += 1

        rank = None

        for r, i in enumerate(ranking[:5], start=1):

            if documents.iloc[i]["unique_key"] in positive_keys:
                rank = r
                break

        reciprocal_rank.append(
            1 / rank if rank else 0
        )

    n = len(test)

    return {
        "Recall@1": recall1 / n,
        "Recall@3": recall3 / n,
        "Recall@5": recall5 / n,
        "MRR": np.mean(reciprocal_rank)
    }


def evaluate_hybrid(alpha=0.5):

    recall1 = 0
    recall3 = 0
    recall5 = 0
    reciprocal_rank = []

    for _, row in test.iterrows():

        positive_keys = {
            x["unique_key"]
            for x in row["positive_contexts"]
        }

        query_tokens = word_tokenize(
            row["question"],
            engine="newmm"
        )

        bm25_scores = bm25.get_scores(query_tokens)

        query_embedding = model.encode(
            [row["question"]],
            show_progress_bar=False
        )[0]

        semantic_scores = cosine_similarity(
            [query_embedding],
            doc_embeddings
        )[0]

        bm25_norm = normalize(bm25_scores)
        semantic_norm = normalize(semantic_scores)

        hybrid_scores = (
            alpha * bm25_norm
            + (1 - alpha) * semantic_norm
        )

        ranking = np.argsort(
            hybrid_scores
        )[::-1]

        top1 = ranking[:1]
        top3 = ranking[:3]
        top5 = ranking[:5]

        keys1 = {
            documents.iloc[i]["unique_key"]
            for i in top1
        }

        keys3 = {
            documents.iloc[i]["unique_key"]
            for i in top3
        }

        keys5 = {
            documents.iloc[i]["unique_key"]
            for i in top5
        }

        if positive_keys & keys1:
            recall1 += 1

        if positive_keys & keys3:
            recall3 += 1

        if positive_keys & keys5:
            recall5 += 1

        rank = None

        for r, i in enumerate(ranking[:5], start=1):

            if documents.iloc[i]["unique_key"] in positive_keys:
                rank = r
                break

        reciprocal_rank.append(
            1 / rank if rank else 0
        )

    n = len(test)

    return {
        "Recall@1": recall1 / n,
        "Recall@3": recall3 / n,
        "Recall@5": recall5 / n,
        "MRR": np.mean(reciprocal_rank)
    }


print("\nRunning BM25...")
bm25_result = evaluate_bm25()

print("\nRunning Semantic Search...")
semantic_result = evaluate_semantic()

print("\nRunning Hybrid Search alpha=0.5...")
hybrid_result = evaluate_hybrid(alpha=0.5)

results = pd.DataFrame([
    {
        "Method": "BM25",
        **bm25_result
    },
    {
        "Method": "Semantic",
        **semantic_result
    },
    {
        "Method": "Hybrid α=0.5",
        **hybrid_result
    }
])

print("\n" + "=" * 75)
print("FINAL TEST RESULTS")
print("=" * 75)

print(
    results.to_string(
        index=False,
        formatters={
            "Recall@1": "{:.4f}".format,
            "Recall@3": "{:.4f}".format,
            "Recall@5": "{:.4f}".format,
            "MRR": "{:.4f}".format
        }
    )
)

results.to_csv(
    "../../results/final_test_results.csv",
    index=False
)

print("\nSaved: ../../results/final_test_results.csv")