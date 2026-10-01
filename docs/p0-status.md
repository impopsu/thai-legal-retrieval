# P0 Status

## Validation benchmark

All methods below were evaluated on the same validation split of 1,643
questions, using the same processed document index and Recall/MRR definitions.

| Method | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---|---:|---:|---:|---:|
| BM25 | 0.4948 | 0.6519 | 0.7158 | 0.5783 |
| Semantic MiniLM | 0.3214 | 0.4729 | 0.5508 | 0.4061 |
| Hybrid MiniLM (alpha=0.5) | 0.5673 | 0.7206 | 0.7669 | 0.6462 |
| BGE-M3 | 0.6543 | 0.8113 | 0.8673 | 0.7379 |
| Hybrid BGE-M3 (alpha=0.5) | 0.6409 | 0.7815 | 0.8411 | 0.7166 |

## Interpretation

- BGE-M3 is currently the strongest retrieval candidate on validation.
- Hybrid MiniLM improves over BM25 while using a much smaller model.
- Hybrid BGE-M3 with alpha `0.5` did not improve over BGE-M3 alone.
- No final method should be selected from validation alone without recording
  the runtime and memory trade-off.

## Reranker validation

The reranker was evaluated on all 1,643 validation questions using Hybrid
MiniLM, alpha `0.5`, candidate top-20 and the
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` model:

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---|---:|---:|---:|---:|
| Hybrid MiniLM baseline | 0.5673 | 0.7206 | 0.7669 | 0.6462 |
| Hybrid MiniLM + reranker | 0.6999 | 0.8223 | 0.8497 | 0.7618 |

The reranker improved every reported retrieval metric on the full validation
split, so it was selected for the final test configuration.

## Final test configuration

```text
Retriever: Hybrid MiniLM
Alpha: 0.5 (selected on validation)
Candidates: top-20
Reranker: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
Final evidence: top-5
```

The configuration was evaluated once on the untouched official test split of
3,742 questions:

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---|---:|---:|---:|---:|
| Hybrid MiniLM baseline | 0.5882 | 0.7435 | 0.7990 | 0.6701 |
| Hybrid MiniLM + reranker | 0.7194 | 0.8354 | 0.8626 | 0.7790 |

The BGE-M3 validation result remains a comparison candidate. It was not used
to tune the final test configuration, and no additional test tuning was done.

## Demo limitation

The final CLI pipeline is wired correctly, but an interactive query can still
retrieve weak evidence even when aggregate benchmark metrics are strong. For
example, the informal query `ถ้าขโมยของคนอื่น มีความผิดอะไร` did not return a
clearly relevant theft provision in the smoke test. This is recorded as an
error-analysis and corpus-coverage issue; the system must not generate a legal
answer when its evidence is not clearly relevant.

## Context retrieval metrics

Context quality is evaluated separately from answer quality by comparing
retrieved `unique_key` values with each row's `positive_contexts`. On a
validation sample of 100 questions using the final Hybrid MiniLM + reranker
pipeline:

| Metric | Score |
|---|---:|
| ContextRecall@1 | 0.6900 |
| ContextRecall@3 | 0.8000 |
| ContextRecall@5 | 0.8300 |
| ContextPrecision@5 | 0.2060 |
| ContextMRR@5 | 0.7498 |

The sample result is a preliminary context-quality report. The full retrieval
benchmark remains the primary aggregate result, and answer F1 must be computed
separately after generated LLM predictions exist.

## Commands

Validation benchmark:

```bash
python scripts/benchmark_retrievers.py \
  --split validation \
  --methods bm25 minilm hybrid_minilm bge hybrid_bge \
  --alpha 0.5
```

Reranker validation:

```bash
python scripts/evaluate_reranker.py \
  --model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
  --retriever hybrid \
  --alpha 0.5
```