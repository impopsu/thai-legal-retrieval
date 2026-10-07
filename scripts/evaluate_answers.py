import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_qa.answer_evaluation import evaluate_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated legal QA answers")
    parser.add_argument("--predictions", required=True, help="JSONL with question and answer")
    parser.add_argument("--references", default="data/raw/test-00000-of-00001.parquet")
    parser.add_argument("--output", default="results/answer_metrics.csv")
    args = parser.parse_args()

    references = pd.read_parquet(args.references)
    reference_by_question = {
        row["question"]: row.to_dict()
        for _, row in references.iterrows()
    }
    latest_predictions = {}
    skipped = Counter()
    total_records = 0
    with Path(args.predictions).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            total_records += 1
            try:
                prediction = json.loads(line)
            except json.JSONDecodeError as error:
                skipped["malformed_json"] += 1
                print(f"skip line {line_number}: malformed JSON ({error})")
                continue
            if not isinstance(prediction, dict):
                skipped["malformed_row"] += 1
                print(f"skip line {line_number}: record is not a JSON object")
                continue
            question = prediction.get("question")
            if not isinstance(question, str) or not question.strip():
                skipped["missing_question"] += 1
                print(f"skip line {line_number}: missing question key")
                continue
            if question not in reference_by_question:
                skipped["unmatched_reference"] += 1
                print(f"skip line {line_number}: no matching reference")
                continue
            if prediction.get("status", "success") != "success":
                skipped["unsuccessful_prediction"] += 1
                print(
                    f"skip line {line_number}: status={prediction.get('status')}"
                )
                continue
            if not isinstance(prediction.get("answer"), str) or not prediction["answer"].strip():
                skipped["missing_answer"] += 1
                print(f"skip line {line_number}: missing answer")
                continue
            latest_predictions[question] = prediction

    scores = []
    for question, prediction in latest_predictions.items():
        try:
            score = evaluate_answer(prediction, reference_by_question[question])
        except Exception as error:
            skipped["evaluation_error"] += 1
            print(f"skip question {question!r}: evaluation failed ({error})")
            continue
        scores.append({"question": question, **score})

    metric_columns = [
        "answer_token_f1",
        "rouge1",
        "rouge2",
        "rougeL",
        "citation_correct",
        "abstained",
    ]
    dataframe = pd.DataFrame(scores, columns=["question", *metric_columns])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    print(f"total_records={total_records}")
    print(f"evaluated={len(dataframe)}")
    print(f"skipped={sum(skipped.values())}")
    for reason, count in sorted(skipped.items()):
        print(f"skipped_{reason}={count}")
    if not dataframe.empty:
        print(dataframe[metric_columns].mean().to_string())
    else:
        print("No valid predictions to evaluate")
    print(f"บันทึกผลแล้ว: {output_path}")


if __name__ == "__main__":
    main()