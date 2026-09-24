import argparse
import json
import os
import time
from pathlib import Path

from google import genai
from google.genai import types


def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_existing(path):
    if not Path(path).exists():
        return set()

    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    if item.get("status") == "success":
                        done.add(item["question"])
                except json.JSONDecodeError:
                    pass
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/qa_prompts_validation.jsonl")
    parser.add_argument("--output", default="results/qa_predictions.jsonl")
    parser.add_argument("--model", default="gemini-3.6-flash")
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=api_key)

    prompts = load_prompts(args.input)
    done = load_existing(args.output)

    print(f"พบ prompts: {len(prompts)}")
    print(f"ทำเสร็จแล้ว: {len(done)}")
    print(f"เหลือ: {len(prompts) - len(done)}")

    with open(args.output, "a", encoding="utf-8") as out:
        for i, item in enumerate(prompts, 1):
            question = item["question"]

            if question in done:
                continue

            print(f"[{i}/{len(prompts)}] {question[:70]}")

            started = time.time()
            answer = None
            error = None

            for attempt in range(1, args.max_retries + 1):
                try:
                    response = client.models.generate_content(
                        model=args.model,
                        contents=item["prompt"],
                        config=types.GenerateContentConfig(
                            temperature=0.1,
                            max_output_tokens=800,
                        ),
                    )

                    answer = response.text.strip()
                    break

                except Exception as exc:
                    error = str(exc)
                    print(f"  attempt {attempt}/{args.max_retries} failed: {error[:200]}")

                    if attempt < args.max_retries:
                        time.sleep(2 ** attempt)

            result = {
                "question": question,
                "answer": answer or "",
                "model": args.model,
                "sources": item.get("sources", []),
                "status": "success" if answer else "error",
                "latency_ms": round((time.time() - started) * 1000),
            }

            if error and not answer:
                result["error"] = error

            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()

            if not answer:
                print("  ข้ามข้อนี้และไปข้อถัดไป")

    print(f"\nเสร็จแล้ว: {args.output}")


if __name__ == "__main__":
    main()
