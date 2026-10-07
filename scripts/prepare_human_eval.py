import argparse
import json
import random
from pathlib import Path

import pandas as pd


def _read_successful_predictions(path):
    predictions = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            prediction = json.loads(line)
            if prediction.get("status") == "success":
                predictions.append(prediction)
    return predictions


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _build_review_row(split, prediction, reference, documents):
    contexts = reference.get("positive_contexts")
    if contexts is None:
        contexts = []
    else:
        contexts = list(contexts)
    positive_contexts = [
        {
            "unique_key": str(item.get("unique_key", "")),
            "law_title": item.get("metadata", {}).get("law_title"),
            "section": item.get("metadata", {}).get("section"),
            "context": item.get("context", ""),
        }
        for item in contexts
    ]
    retrieved_contexts = []
    for source in prediction.get("sources", []):
        source_key = str(source.get("unique_key", ""))
        retrieved_contexts.append({
            **source,
            "context": documents.get(source_key, ""),
        })
    return {
        "split": split,
        "question": prediction["question"],
        "model": prediction.get("model"),
        "model_answer": prediction.get("answer", ""),
        "reference_answer": reference.get("positive_answer", ""),
        "reference_positive_contexts": positive_contexts,
        "retrieved_sources": retrieved_contexts,
        "scores": {
            "correctness": None,
            "faithfulness": None,
            "citation": None,
            "completeness": None,
            "abstention": None,
        },
        "reviewer_id": "",
        "notes": "",
    }


def build_human_eval_rows(
    validation_predictions,
    validation_references,
    test_predictions,
    test_references,
    documents,
    sample_per_split=20,
    seed=42,
):
    rng = random.Random(seed)
    output = []
    for split, predictions, references in (
        ("validation", validation_predictions, validation_references),
        ("test", test_predictions, test_references),
    ):
        reference_by_question = {
            row["question"]: row for row in references.to_dict("records")
        }
        matched = [
            prediction
            for prediction in predictions
            if prediction.get("question") in reference_by_question
        ]
        if len(matched) < sample_per_split:
            raise ValueError(
                f"Only {len(matched)} matched {split} predictions; "
                f"need {sample_per_split}"
            )
        for prediction in rng.sample(matched, sample_per_split):
            reference = reference_by_question[prediction["question"]]
            output.append(
                _build_review_row(split, prediction, reference, documents)
            )
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a blinded human-review sample for generated answers"
    )
    parser.add_argument(
        "--validation-predictions",
        default="results/qa_predictions_validation_100.jsonl",
    )
    parser.add_argument(
        "--validation-references",
        default="data/processed/validation_retrieval.parquet",
    )
    parser.add_argument(
        "--test-predictions",
        default="results/qa_predictions_test_100.jsonl",
    )
    parser.add_argument(
        "--test-references",
        default="data/raw/test-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--documents",
        default="data/processed/legal_documents_categorized.csv",
    )
    parser.add_argument("--sample-per-split", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="results/human_eval_answers_rubric.jsonl",
    )
    args = parser.parse_args()

    document_frame = pd.read_csv(args.documents, encoding="utf-8-sig")
    document_text = {
        str(row.unique_key): row.context
        for row in document_frame.itertuples(index=False)
    }
    rows = build_human_eval_rows(
        _read_successful_predictions(args.validation_predictions),
        pd.read_parquet(args.validation_references),
        _read_successful_predictions(args.test_predictions),
        pd.read_parquet(args.test_references),
        document_text,
        sample_per_split=args.sample_per_split,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, default=_json_default))
            file.write("\n")
    print(f"Prepared {len(rows)} review rows at {output_path}")


if __name__ == "__main__":
    main()