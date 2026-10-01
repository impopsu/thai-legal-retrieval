import argparse
import json
import os
import random
import time
from pathlib import Path


DEFAULT_MODEL = "gemini-3.1-pro-preview"
TRANSIENT_MARKERS = (
    "429", "500", "502", "503", "504", "rate limit",
    "resource exhausted", "temporarily unavailable", "timeout", "timed out",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate grounded Thai legal answers with Gemini"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--base-delay", type=float, default=2.0)
    parser.add_argument("--retry-errors", action="store_true")
    return parser.parse_args()


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    records = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                if record.get("question"):
                    records[record["question"]] = record
    return records


def is_transient(error: Exception) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in TRANSIENT_MARKERS)


def safe_error(error: Exception, api_key: str) -> str:
    return str(error).replace(api_key, "[REDACTED]")[:1000]


def generate_answer(client, model: str, prompt: str, max_output_tokens: int) -> str:
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=max_output_tokens,
        ),
    )
    answer = getattr(response, "text", None)
    if not answer or not answer.strip():
        raise RuntimeError("Gemini returned an empty response")
    return answer.strip()


def main() -> None:
    args = parse_args()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Set it in the terminal environment; "
            "the key is never accepted as a command-line argument."
        )

    from google import genai

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing(output_path)
    client = genai.Client(api_key=api_key)
    input_count = success_count = error_count = 0
    started = time.monotonic()

    with input_path.open(encoding="utf-8") as input_file, output_path.open(
        "a", encoding="utf-8"
    ) as output_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            input_count += 1
            prompt_record = json.loads(line)
            question = prompt_record.get("question")
            prompt = prompt_record.get("prompt")
            if not question or not prompt:
                raise ValueError(f"Input line {line_number} lacks question or prompt")

            previous = existing.get(question)
            if previous and (
                previous.get("status") == "success" or not args.retry_errors
            ):
                if previous.get("status") == "success":
                    success_count += 1
                else:
                    error_count += 1
                continue

            record = {
                "question": question,
                "answer": None,
                "model": args.model,
                "category": prompt_record.get("category", "all"),
                "sources": prompt_record.get("sources", []),
                "status": "error",
                "error": None,
                "attempts": 0,
                "latency_ms": None,
            }
            request_started = time.monotonic()
            for attempt in range(1, args.max_retries + 2):
                record["attempts"] = attempt
                try:
                    record["answer"] = generate_answer(
                        client, args.model, prompt, args.max_output_tokens
                    )
                    record["status"] = "success"
                    record["error"] = None
                    success_count += 1
                    break
                except Exception as error:
                    record["error"] = {
                        "type": type(error).__name__,
                        "message": safe_error(error, api_key),
                    }
                    if attempt > args.max_retries or not is_transient(error):
                        error_count += 1
                        break
                    delay = args.base_delay * (2 ** (attempt - 1))
                    time.sleep(delay + random.uniform(0, args.base_delay))

            record["latency_ms"] = round(
                (time.monotonic() - request_started) * 1000, 2
            )
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()

    print(f"input_records={input_count}")
    print(f"success={success_count}")
    print(f"errors={error_count}")
    print(f"runtime_seconds={time.monotonic() - started:.2f}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
