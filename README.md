# Thai Legal QA

Thai legal retrieval and grounded question answering using BM25, multilingual
semantic search and hybrid retrieval.

## Current status

The reusable core is in `legal_qa/`. Original experiments remain in `experiments/`
for comparison. The default retrieval convention is:

```text
hybrid_score = alpha * normalized_bm25 + (1 - alpha) * normalized_semantic
```

Answer generation is optional and only loads a language model when one is
explicitly supplied.

## Run a retrieval evaluation

```bash
python scripts/evaluate_retrieval.py --methods bm25
```

To evaluate the hybrid retriever and cache MiniLM document embeddings:

```bash
python scripts/evaluate_retrieval.py --methods bm25 hybrid --alpha 0.5
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
