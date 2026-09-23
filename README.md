# Thai Legal QA

Thai legal retrieval and grounded question answering using BM25, multilingual
semantic search, hybrid retrieval and reranking.

## Current status

The reusable core is in `legal_qa/`. Original experiments remain in `experiments/`
for comparison. No retrieval method is selected as the final production method
yet; BM25, Hybrid MiniLM, BGE-M3 and Hybrid BGE-M3 must be compared under the
same evaluation protocol first. The hybrid score convention is:

```text
hybrid_score = alpha * normalized_bm25 + (1 - alpha) * normalized_semantic
```

Answer generation is optional and only loads a language model when one is
explicitly supplied.

## Run a retrieval evaluation

```bash
python scripts/evaluate_retrieval.py --methods bm25
```

Compare retrievers on validation using one protocol:

```bash
python scripts/benchmark_retrievers.py \
	--split validation \
	--methods bm25 minilm hybrid_minilm
```

To evaluate the hybrid retriever and cache MiniLM document embeddings:

```bash
python scripts/evaluate_retrieval.py --methods bm25 hybrid --alpha 0.5
```

Evaluate an optional cross-encoder reranker on validation data:

```bash
python scripts/evaluate_reranker.py \
	--model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

## Run the QA demo

```bash
python scripts/run_qa_demo.py
```

Without a generator model, the demo displays ranked legal evidence and the
grounded prompt. Pass `--generator-model` to use a compatible Hugging Face
text-to-text model for answer generation.

See [docs/roadmap.md](docs/roadmap.md) for scope, metrics and next milestones.
See [docs/dataset.md](docs/dataset.md) for dataset source, papers and the
untouched-test evaluation protocol.
See [docs/p0-status.md](docs/p0-status.md) for the current retrieval benchmark
and remaining blockers.
