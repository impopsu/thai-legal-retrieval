from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .models import SearchResult


def build_prompt(question: str, results: Sequence[SearchResult]) -> str:
    evidence = "\n\n".join(
        f"[เอกสารที่ {rank}]\nกฎหมาย: {result.document.law_title}\n"
        f"มาตรา: {result.document.section}\nเนื้อหา: {result.document.context}"
        for rank, result in enumerate(results, start=1)
    )
    return f"""คุณเป็นผู้ช่วยตอบคำถามกฎหมายไทย
ตอบโดยใช้ข้อมูลจากเอกสารอ้างอิงด้านล่างเท่านั้น
ห้ามสร้างชื่อกฎหมาย เลขมาตรา หรือรายละเอียดที่ไม่มีอยู่ในเอกสาร
ถ้าข้อมูลไม่เพียงพอ ให้ตอบว่า "ไม่พบข้อมูลเพียงพอจากเอกสารที่ค้นได้"
ให้ระบุชื่อกฎหมายและมาตราที่ใช้อ้างอิงท้ายคำตอบ

คำถาม:
{question}

เอกสารอ้างอิง:
{evidence}
"""


class Generator(Protocol):
    def generate(self, question: str, results: Sequence[SearchResult]) -> str:
        ...


class TransformersGenerator:
    def __init__(self, model_name: str, max_new_tokens: int = 256):
        from transformers import pipeline

        self.max_new_tokens = max_new_tokens
        self.generator = pipeline("text2text-generation", model=model_name)

    def generate(self, question: str, results: Sequence[SearchResult]) -> str:
        output = self.generator(
            build_prompt(question, results),
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        return output[0]["generated_text"].strip()


class LegalQAPipeline:
    def __init__(self, retriever, reranker=None, generator: Generator | None = None):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

    def answer(self, question: str, top_k: int = 5, candidate_k: int | None = None):
        candidate_k = candidate_k or max(top_k, top_k * 4)
        sources = self.retriever.search(question, top_k=candidate_k)
        if self.reranker:
            sources = self.reranker.rerank(question, sources, top_k)
        else:
            sources = sources[:top_k]
        prompt = build_prompt(question, sources)
        answer = self.generator.generate(question, sources) if self.generator else None
        return QAResponse(question, answer, sources, prompt)


@dataclass(frozen=True)
class QAResponse:
    question: str
    answer: str | None
    sources: list[SearchResult]
    prompt: str