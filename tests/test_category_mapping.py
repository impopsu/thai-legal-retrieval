import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_qa import HybridRetriever, LegalQAPipeline, load_documents
from legal_qa.categories import classify_category
from legal_qa.data import load_documents
from legal_qa.reranking import CrossEncoderReranker


def test_category_mapping_uses_law_code_and_section():
    dataset = Path("data/processed/legal_documents_categorized.csv")
    assert classify_category(
        law_code="ธ0012-1B-0001",
        section="10",
        law_title="พระราชบัญญัติธุรกิจสถาบันการเงิน พ.ศ. 2551",
        dataset_path=dataset,
    ) == "financial"
    assert classify_category(
        law_code="ก0039-1B-0002",
        section="1",
        law_title="พระราชบัญญัติกำหนดความผิดเกี่ยวกับห้างหุ้นส่วนจดทะเบียน",
        dataset_path=dataset,
    ) == "company"


def test_category_mapping_is_not_question_or_context_dependent():
    dataset = Path("data/processed/legal_documents_categorized.csv")
    law_title = "พระราชบัญญัติธุรกิจสถาบันการเงิน พ.ศ. 2551"
    assert classify_category(
        "ถ้าหลักทรัพย์มีความผิดอะไร",
        "ธนาคาร",
        law_title=law_title,
        section="20",
        dataset_path=dataset,
    ) == "financial"
    assert classify_category(
        "เรื่องเกี่ยวกับบริษัทและหุ้น",
        "คำถามเกี่ยวกับการฟ้องร้อง",
        law_title=law_title,
        section="20",
        dataset_path=dataset,
    ) == "financial"


def test_load_documents_keeps_rows_and_uses_structural_fields(tmp_path):
    csv_path = tmp_path / "docs.csv"
    pd.DataFrame(
        [
            {
                "unique_key": "A-1",
                "law_code": "ธ0012-1B-0001",
                "law_title": "พระราชบัญญัติธุรกิจสถาบันการเงิน พ.ศ. 2551",
                "section": "10",
                "context": "เรื่องธนาคาร",
            },
            {
                "unique_key": "B-1",
                "law_code": "ก0039-1B-0002",
                "law_title": "พระราชบัญญัติกำหนดความผิดเกี่ยวกับห้างหุ้นส่วนจดทะเบียน",
                "section": "1",
                "context": "เรื่องบริษัท",
            },
        ]
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")

    documents = load_documents(csv_path)
    assert len(documents) == 2
    assert {document.category for document in documents} == {"financial", "company"}


def test_selected_category_filters_before_retrieval_and_rerank():
    documents = load_documents("data/processed/legal_documents_categorized.csv")
    category = "company"
    query = "กฎหมาย"

    filtered = [doc for doc in documents if doc.category == category]
    assert len(filtered) > 20

    retriever = HybridRetriever(
        documents,
        alpha=0.5,
        embeddings_path="results/minilm_document_embeddings.pt",
    )
    all_results = retriever.search(query, top_k=20, category="all")
    assert len(all_results) == 20
    assert any(result.document.category != category for result in all_results)

    filtered_results = retriever.search(query, top_k=20, category=category)
    assert len(filtered_results) == 20
    assert all(result.document.category == category for result in filtered_results)

    reranker = CrossEncoderReranker("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    pipeline = LegalQAPipeline(retriever, reranker=reranker)
    response = pipeline.answer(query, top_k=5, candidate_k=20, category=category)
    financial_results = retriever.search(query, top_k=20, category="financial")

    assert len(response.sources) == 5
    assert all(result.document.category == category for result in response.sources)
    assert all(result.document.category == category for result in filtered_results)
    assert {result.document.unique_key for result in filtered_results} & {
        result.document.unique_key for result in financial_results
    } == set()
