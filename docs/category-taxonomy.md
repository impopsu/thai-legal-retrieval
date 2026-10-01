# Legal Category Taxonomy

The current dataset does not provide a category column, so the preparation
script assigns an extensible rule-based category from the question, law title
and context. The category is a retrieval filter, not a gold legal annotation.

Current categories:

```text
property   กฎหมายทรัพย์สิน
criminal   กฎหมายอาญา
murder     กฎหมายฆาตกรรม/ชีวิตและร่างกาย
financial  กฎหมายการเงิน
company    กฎหมายบริษัทและนิติบุคคล
labor      กฎหมายแรงงาน
family     กฎหมายครอบครัวและมรดก
tax        กฎหมายภาษีอากร
procedure  กฎหมายวิธีพิจารณา
other      กฎหมายหมวดอื่น
```

Add or edit keyword rules in `legal_qa/categories.py` when a new legal area is
needed. For higher-quality research results, manually review category labels or
replace the heuristic classifier with an annotated category field.

Prepare the derived files with:

```bash
python scripts/prepare_qa_dataset.py
```

This creates:

```text
data/processed/qa_records.parquet
data/processed/legal_documents_categorized.csv
```