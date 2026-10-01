import re
from collections import Counter
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


def rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    prediction_tokens = answer_tokens(prediction)
    reference_tokens = answer_tokens(reference)

    def fmeasure(overlap: int, prediction_count: int, reference_count: int) -> float:
        if not prediction_count or not reference_count:
            return float(prediction_count == reference_count)
        precision = overlap / prediction_count
        recall = overlap / reference_count
        return 2 * precision * recall / (precision + recall) if overlap else 0.0

    def ngram_fmeasure(n: int) -> float:
        prediction_ngrams = Counter(
            tuple(prediction_tokens[index:index + n])
            for index in range(len(prediction_tokens) - n + 1)
        )
        reference_ngrams = Counter(
            tuple(reference_tokens[index:index + n])
            for index in range(len(reference_tokens) - n + 1)
        )
        overlap = sum((prediction_ngrams & reference_ngrams).values())
        return fmeasure(overlap, sum(prediction_ngrams.values()), sum(reference_ngrams.values()))

    lcs = [0] * (len(reference_tokens) + 1)
    for prediction_token in prediction_tokens:
        previous = 0
        for index, reference_token in enumerate(reference_tokens, start=1):
            current = lcs[index]
            if prediction_token == reference_token:
                lcs[index] = previous + 1
            else:
                lcs[index] = max(lcs[index], lcs[index - 1])
            previous = current

    return {
        "rouge1": ngram_fmeasure(1),
        "rouge2": ngram_fmeasure(2),
        "rougeL": fmeasure(lcs[-1], len(prediction_tokens), len(reference_tokens)),
    }


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
        **rouge_scores(answer, reference),
        "citation_correct": float(
            citation_correct(answer, reference_row["positive_contexts"])
        ),
        "abstained": float(ABSTENTION_PHRASE in answer),
    }