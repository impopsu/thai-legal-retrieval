import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_qa.categories import classify_category


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare question, positive context and category records"
    )
    parser.add_argument("--train", default="data/raw/train-00000-of-00001.parquet")
    parser.add_argument("--test", default="data/raw/test-00000-of-00001.parquet")
    parser.add_argument("--output", default="data/processed/qa_records.parquet")
    parser.add_argument(
        "--documents-output",
        default="data/processed/legal_documents_categorized.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    documents = {}

    for split, path in (("train", args.train), ("test", args.test)):
        dataframe = pd.read_parquet(path)
        for row_index, row in dataframe.iterrows():
            for context_index, item in enumerate(row["positive_contexts"]):
                metadata = item["metadata"]
                category = classify_category(
                    row["question"],
                    metadata["law_title"],
                    item["context"],
                )
                records.append({
                    "question_id": f"{split}-{row_index}",
                    "split": split,
                    "question": row["question"],
                    "positive_context_index": context_index,
                    "positive_context_key": item["unique_key"],
                    "positive_context": item["context"],
                    "positive_answer": row["positive_answer"],
                    "law_code": metadata["law_code"],
                    "law_title": metadata["law_title"],
                    "section": metadata["section"],
                    "category": category,
                })
                documents.setdefault(item["unique_key"], {
                    "unique_key": item["unique_key"],
                    "law_code": metadata["law_code"],
                    "law_title": metadata["law_title"],
                    "section": metadata["section"],
                    "context": item["context"],
                    "category": category,
                })

    records_path = Path(args.output)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(records_path, index=False)

    documents_path = Path(args.documents_output)
    documents_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(documents.values()).to_csv(
        documents_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"QA records: {len(records)} -> {records_path}")
    print(f"Documents: {len(documents)} -> {documents_path}")
    print("Categories:")
    print(pd.DataFrame(records)["category"].value_counts().to_string())


if __name__ == "__main__":
    main()