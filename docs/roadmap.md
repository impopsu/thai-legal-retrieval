# Project Roadmap

## Current scope

The project has a reusable retrieval core in `legal_qa/` and keeps the original
scripts in `experiments/` for reproducible comparisons. The main convention is:

```text
hybrid_score = alpha * normalized_bm25 + (1 - alpha) * normalized_semantic
```

No retrieval method is selected as the final system yet. BM25, Hybrid MiniLM,
BGE-M3 and Hybrid BGE-M3 must be benchmarked under the same validation/test
protocol before choosing the main retriever. Runtime and memory are reported
alongside retrieval quality because BGE-M3 is expensive on CPU.

## Milestones

1. Retrieval benchmark: compare BM25, Hybrid MiniLM, BGE-M3 and Hybrid BGE-M3.
2. Reranker benchmark: rerank the same candidate set and compare against the
	selected retrieval baselines on validation.
3. Final method selection: choose retrieval and reranker using validation only.
4. Evidence pipeline: return ranked legal documents with law title and section.
5. Grounded QA: generate an answer only from retrieved evidence and include citations.
6. QA evaluation: measure answer correctness, citation correctness and faithfulness.
7. Demo: expose the selected pipeline through a CLI or web interface.

Reranking is a core experiment for improving the basic similarity-based
retrieval. It must be evaluated on validation before it is included in a final
test configuration; it is not treated as optional by design.

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