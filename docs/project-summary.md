# Thai Legal QA / RAG Project Summary

## 1. Project Goal

ระบบรับคำถามกฎหมายภาษาไทย ค้นหา context กฎหมายที่เกี่ยวข้อง จัดอันดับด้วย
reranker และเตรียม evidence สำหรับ LLM โดยคำตอบต้องอ้างอิงจาก evidence เท่านั้น

```text
Question
-> Hybrid Retrieval
-> Top-20 candidates
-> Cross-Encoder Reranker
-> Top-5 evidence
-> Grounded LLM prompt
-> Answer + citation
```

## 2. Dataset

ชื่อ: **WangchanX-Legal-ThaiCCL-RAG**

Source: https://huggingface.co/datasets/airesearch/WangchanX-Legal-ThaiCCL-RAG

Papers:

- NitiBench: https://aclanthology.org/2025.emnlp-main.1739/
- A Free Format Legal Question Answering System: https://aclanthology.org/2021.nllp-1.11/

Local data:

```text
data/raw/train-00000-of-00001.parquet
data/raw/test-00000-of-00001.parquet
```

Current split:

```text
Train: 8,211 questions
Validation: 1,643 questions
Test: 3,742 questions
```

Each record contains `question`, `positive_contexts`,
`hard_negative_contexts`, `positive_answer` and `hard_negative_answer`.

Prepare question-level records and categorized documents:

```bash
python scripts/prepare_qa_dataset.py
```

Outputs:

```text
data/processed/qa_records.parquet
data/processed/legal_documents_categorized.csv
```

The QA records contain question, positive context, positive answer, law
metadata and category, one row per positive context.

## 3. Legal Categories

The dataset has no official category field. The project uses an extensible
rule-based classifier from question, law title and context. It is a retrieval
filter, not a gold annotation.

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

Rules can be extended in `legal_qa/categories.py`.

Category demo:

```bash
python scripts/run_qa_demo.py --category property
python scripts/run_web_demo.py --documents data/processed/legal_documents_categorized.csv
```

## 4. Final Retrieval Configuration

```text
Retriever: Hybrid MiniLM
Alpha: 0.5
Candidates: top-20
Reranker: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
Final evidence: top-5
```

The official test split is not used for selecting alpha, top-k, model or
reranker.

## 5. Retrieval Results

Full validation benchmark:

| Method | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---|---:|---:|---:|---:|
| BM25 | 0.4948 | 0.6519 | 0.7158 | 0.5783 |
| Semantic MiniLM | 0.3214 | 0.4729 | 0.5508 | 0.4061 |
| Hybrid MiniLM | 0.5673 | 0.7206 | 0.7669 | 0.6462 |
| BGE-M3 | 0.6543 | 0.8113 | 0.8673 | 0.7379 |
| Hybrid BGE-M3 | 0.6409 | 0.7815 | 0.8411 | 0.7166 |

Full validation reranker result:

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---|---:|---:|---:|---:|
| Hybrid MiniLM baseline | 0.5673 | 0.7206 | 0.7669 | 0.6462 |
| Hybrid MiniLM + reranker | 0.6999 | 0.8223 | 0.8497 | 0.7618 |

Untouched test retrieval result:

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---|---:|---:|---:|---:|
| Hybrid MiniLM baseline | 0.5882 | 0.7435 | 0.7990 | 0.6701 |
| Hybrid MiniLM + reranker | 0.7194 | 0.8354 | 0.8626 | 0.7790 |

## 6. Context Evaluation

Retrieved context is compared with `positive_contexts` using document
`unique_key`.

Run:

```bash
python scripts/evaluate_context.py \
  --questions data/processed/validation_retrieval.parquet \
  --documents data/processed/legal_documents_categorized.csv \
  --limit 100
```

Validation sample result, 100 questions:

```text
ContextRecall@1:     0.6900
ContextRecall@3:     0.8000
ContextRecall@5:     0.8300
ContextPrecision@5:  0.2060
ContextMRR@5:        0.7498
```

## 7. Gemini Answer Evaluation

The grounded prompt is generated from the final top-5 evidence. Answers were
generated with:

```text
Model: gemini-3.5-flash-lite
Temperature: 0.0
```

Validation 100 questions:

```text
Answer Token F1:  0.395647
Citation Correct: 0.640000
Abstained:        0.280000
```

Untouched test sample, 100 questions:

```text
Answer Token F1:  0.487709
Citation Correct: 0.800000
Abstained:        0.130000
```

Generate/evaluate commands:

```bash
python scripts/export_qa_prompts.py \
  --questions data/processed/validation_retrieval.parquet \
  --limit 100 \
  --output results/qa_prompts_validation_100.jsonl

python scripts/generate_gemini_answers.py \
  --input results/qa_prompts_validation_100.jsonl \
  --output results/qa_predictions_validation_100.jsonl \
  --model gemini-3.5-flash-lite \
  --retry-errors

python scripts/evaluate_answers.py \
  --predictions results/qa_predictions_validation_100.jsonl \
  --references data/processed/validation_retrieval.parquet \
  --output results/answer_metrics_validation_100.csv
```

The generator supports retry/backoff, resume and error logging. API keys are
read only from `GEMINI_API_KEY` and are never stored in the repository.

## 8. Human Evaluation

For 50-100 generated answers, reviewers can score each item from 0 to 2:

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| Answer correctness | Wrong | Partly correct | Correct |
| Evidence faithfulness | Unsupported | Partly supported | Fully supported |
| Citation correctness | Wrong/missing | Partly correct | Correct law and section |
| Completeness | Incomplete | Partly complete | Complete |
| Abstention behavior | Hallucinates | Unclear | Correctly refuses/qualifies |

Recommended columns:

```text
question, answer, retrieved_sources, answer_correctness,
evidence_faithfulness, citation_correctness, completeness,
abstention_behavior, notes, reviewer_id
```

## 9. Demo and Hardware

CLI:

```bash
python scripts/run_qa_demo.py
```

Web:

```bash
python scripts/run_web_demo.py \
  --documents data/processed/legal_documents_categorized.csv
```

Hardware:

```bash
python scripts/check_hardware.py
```

The current environment has no NVIDIA CUDA GPU and uses CPU fallback.

## 10. Limitations

- Category labels are rule-based, not gold annotations.
- Answer evaluation uses 100-question validation and test samples.
- Some informal questions still retrieve weak evidence despite good aggregate
  metrics; the system should abstain when evidence is not clearly relevant.
- Human expert evaluation is still recommended for a stronger legal QA claim.

## 11. Report Structure

1. Problem definition
2. Dataset and related work
3. Retrieval and reranking methods
4. Validation/test protocol
5. Retrieval and answer results
6. Category-filtered retrieval
7. Error analysis and limitations
8. Conclusion
