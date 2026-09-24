import re
from collections.abc import Iterable

from pythainlp.tokenize import word_tokenize


ABSTENTION_PHRASE = "ไม่พบข้อมูลเพียงพอจากเอกสารที่ค้นได้"


def answer_tokens(text: str) -> list[str]:
    tokens = word_tokenize(str(text).lower(), engine="newmm")
    return [token for token in tokens if re.search(r"\w", token, re.UNICODE)]


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = answer_tokens(prediction)
    reference_tokens = answer_tokens(reference)
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)

    overlap = sum(
        min(prediction_tokens.count(token), reference_tokens.count(token))
        for token in set(prediction_tokens)
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def citation_correct(answer: str, positive_contexts: Iterable[dict]) -> bool:
    answer_text = str(answer)
    return any(
        str(context["metadata"]["law_title"]) in answer_text
        and str(context["metadata"]["section"]) in answer_text
        for context in positive_contexts
    )


def evaluate_answer(prediction: dict, reference_row: dict) -> dict[str, float]:
    answer = str(prediction.get("answer", ""))
    reference = str(reference_row.get("positive_answer", ""))
    return {
        "answer_token_f1": token_f1(answer, reference),
        "citation_correct": float(
            citation_correct(answer, reference_row["positive_contexts"])
        ),
        "abstained": float(ABSTENTION_PHRASE in answer),
    }