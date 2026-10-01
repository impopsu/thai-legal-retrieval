# Reproducibility Checklist

## Dataset

- Dataset: WangchanX-Legal-ThaiCCL-RAG
- Source: https://huggingface.co/datasets/airesearch/WangchanX-Legal-ThaiCCL-RAG
- Local train/test files are kept under `data/raw/`.
- The official test split is not used for tuning.

## Retrieval

- Run commands from the repository root.
- Check hardware with `python scripts/check_hardware.py` before choosing CPU/GPU.
- Create validation data with `scripts/split_train_validation.py`.
- Prepare question-level context/category records with `scripts/prepare_qa_dataset.py`.
- Select alpha with `scripts/select_alpha.py`.
- Benchmark methods with `scripts/benchmark_retrievers.py`.
- Evaluate retrieved contexts with `scripts/evaluate_context.py`.
- Record model names, alpha, candidate-k, top-k, metrics and runtime.

## Final configuration

```text
Retriever: Hybrid MiniLM
Alpha: 0.5
Candidate-k: 20
Reranker: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
Evidence top-k: 5
```

## Commands

```bash
python scripts/split_train_validation.py
python scripts/select_alpha.py
python scripts/benchmark_retrievers.py --split validation
python scripts/run_web_demo.py
```

Generated embeddings, datasets and result files are intentionally ignored by
Git. Recreate them with the commands above when setting up a new environment.

## Gemini generator

The answer generator uses model `gemini-3.1-pro-preview` and reads
`GEMINI_API_KEY` only from the environment. It uses temperature `0.0`, retries
transient errors with exponential backoff, resumes successful JSONL records and
records non-transient failures without stopping the whole batch.