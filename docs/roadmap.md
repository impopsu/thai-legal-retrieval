# Project Roadmap

## Current scope

The project has a reusable retrieval core in `legal_qa/` and keeps the original
scripts in `experiments/` for reproducible comparisons. The main convention is:

```text
hybrid_score = alpha * normalized_bm25 + (1 - alpha) * normalized_semantic
```

The default production-friendly retriever is multilingual MiniLM because BGE-M3
is expensive to run on CPU. BGE-M3 remains an experiment and comparison model.

## Milestones

1. Retrieval baseline: BM25, semantic search, hybrid search, Recall@k and MRR@k.
2. Evidence pipeline: return ranked legal documents with law title and section.
3. Grounded QA: generate an answer only from retrieved evidence and include citations.
4. QA evaluation: measure answer correctness, citation correctness and faithfulness.
5. Demo: expose the pipeline through a CLI or web interface.

## Current evaluation commands

Create validation data from the training split only:

```bash
python scripts/split_train_validation.py
```

Select `alpha` using validation:

```bash
python scripts/select_alpha.py
```

Run the final test evaluation with the selected alpha. Do not use this command
to tune parameters:

```bash
python scripts/evaluate_retrieval.py --methods bm25 hybrid --alpha 0.5
```

## Reproducibility notes

- Run scripts from the repository root.
- Use validation to select `alpha`; use test only for final reporting.
- Treat `MRR@5` as a top-five metric, not full-ranking MRR.
- Generated datasets, embeddings and result files are ignored by Git by default.