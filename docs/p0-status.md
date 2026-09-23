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

## Remaining P0 blockers

1. Full reranker validation has not completed on the CPU container. Short smoke
   tests pass, but full cross-encoder inference was terminated by the runtime.
2. BGE-M3 final test evaluation has not been rerun through the unified runner;
   its validation run took about 85 minutes on this CPU environment.
3. The final retrieval plus reranker configuration therefore must not yet be
   claimed as final.

## Reranker pilot

On the first 100 validation questions, using Hybrid MiniLM with the
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` model and candidate top-20:

| Configuration | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---|---:|---:|---:|---:|
| Hybrid MiniLM baseline | 0.5600 | 0.7200 | 0.7500 | 0.6353 |
| Hybrid MiniLM + reranker | 0.6900 | 0.8000 | 0.8300 | 0.7498 |

This is an encouraging pilot only. The reranker must still be evaluated on the
full validation split before it can be selected for the final test run.

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