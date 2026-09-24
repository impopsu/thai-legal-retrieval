import argparse
import json
import sys
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
    scores = []
    with Path(args.predictions).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            prediction = json.loads(line)
            question = prediction.get("question")
            if question not in reference_by_question:
                raise ValueError(
                    f"Prediction line {line_number} has no matching reference question"
                )
            scores.append({
                "question": question,
                **evaluate_answer(prediction, reference_by_question[question]),
            })

    if not scores:
        raise ValueError("Prediction file is empty")
    dataframe = pd.DataFrame(scores)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    print(dataframe.drop(columns="question").mean().to_string())
    print(f"บันทึกผลแล้ว: {output_path}")


if __name__ == "__main__":
    main()