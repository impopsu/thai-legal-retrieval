# Reproducibility Checklist

## Dataset

- Dataset: WangchanX-Legal-ThaiCCL-RAG
- Source: https://huggingface.co/datasets/airesearch/WangchanX-Legal-ThaiCCL-RAG
- Local train/test files are kept under `data/raw/`.
- The official test split is not used for tuning.

## Retrieval

- Run commands from the repository root.
- Create validation data with `scripts/split_train_validation.py`.
- Select alpha with `scripts/select_alpha.py`.
- Benchmark methods with `scripts/benchmark_retrievers.py`.
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