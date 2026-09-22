import pandas as pd

train = pd.read_parquet(
    "data/raw/train-00000-of-00001.parquet"
)

test = pd.read_parquet(
    "data/raw/test-00000-of-00001.parquet"
)

documents = {}

for df in [train, test]:
    for row in df["positive_contexts"]:
        for item in row:
            key = item["unique_key"]

            if key not in documents:
                documents[key] = {
                    "unique_key": key,
                    "law_code": item["metadata"]["law_code"],
                    "law_title": item["metadata"]["law_title"],
                    "section": item["metadata"]["section"],
                    "context": item["context"]
                }

legal_documents = pd.DataFrame(documents.values())

legal_documents.to_csv(
    "data/processed/legal_documents.csv",
    index=False,
    encoding="utf-8-sig"
)

print("สร้าง legal_documents.csv เรียบร้อย")
print("จำนวน documents:", len(legal_documents))
print("\n===== ตัวอย่าง =====")
print(legal_documents.head())