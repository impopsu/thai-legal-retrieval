# Dataset

This project uses **WangchanX-Legal-ThaiCCL-RAG**.

- Source: [Hugging Face: airesearch/WangchanX-Legal-ThaiCCL-RAG](https://huggingface.co/datasets/airesearch/WangchanX-Legal-ThaiCCL-RAG)
- Language: Thai
- Task: Thai legal question answering and retrieval-augmented generation
- Local source of truth: `data/raw/train-00000-of-00001.parquet` and `data/raw/test-00000-of-00001.parquet`
- Train examples in the current files: 8,211
- Test examples in the current files: 3,742

Each record contains:

```text
question
positive_contexts
hard_negative_contexts
positive_answer
hard_negative_answer
```

The current processed document index is generated in
`data/processed/legal_documents.csv` from the context records. Its construction
should be described explicitly in reports because the benchmark labels provide
the contexts used to build this local index.

For question-level experiments, run `scripts/prepare_qa_dataset.py`. It creates
`data/processed/qa_records.parquet`, one row per positive context, with question,
positive context, positive answer, metadata and an extensible rule-based legal
category. It also creates
`data/processed/legal_documents_categorized.csv` for category-filtered retrieval.

## Evaluation protocol

- The official test Parquet remains untouched.
- Only the training split may be divided into train and validation subsets.
- Model, alpha, top-k and reranking settings must be selected on validation.
- The official test split is used once for final reporting.

## Related papers

1. Akarajaradwong et al. **NitiBench: Benchmarking LLM Frameworks on Thai
   Legal Question Answering Capabilities.** EMNLP 2025.
   [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1739/)
2. **A Free Format Legal Question Answering System.** NLLP 2021.
   [ACL Anthology](https://aclanthology.org/2021.nllp-1.11/)