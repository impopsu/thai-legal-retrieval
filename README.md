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

The default demo uses the selected final pipeline: Hybrid MiniLM alpha `0.5`,
candidate top-20 and Cross-Encoder reranking to top-5. Use `--no-reranker` for
a faster baseline. Without a generator model, the demo displays ranked legal
evidence and the grounded prompt. Pass `--generator-model` to use a compatible
Hugging Face text-to-text model for answer generation.

See [docs/roadmap.md](docs/roadmap.md) for scope, metrics and next milestones.
See [docs/dataset.md](docs/dataset.md) for dataset source, papers and the
untouched-test evaluation protocol.
See [docs/p0-status.md](docs/p0-status.md) for the current retrieval benchmark
and remaining blockers.
See [docs/human-evaluation.md](docs/human-evaluation.md) for the answer review
rubric and [docs/reproducibility.md](docs/reproducibility.md) for setup steps.

Prepare question-level positive contexts and categories:

```bash
python scripts/prepare_qa_dataset.py
```

Evaluate retrieved contexts against `positive_contexts`:

```bash
python scripts/evaluate_context.py \
	--questions data/processed/validation_retrieval.parquet \
	--documents data/processed/legal_documents_categorized.csv \
	--limit 100
```

Check GPU availability:

```bash
python scripts/check_hardware.py
```

See [docs/category-taxonomy.md](docs/category-taxonomy.md) for category rules
and limitations.
See [docs/context-metrics.md](docs/context-metrics.md) for the retrieved-context
evaluation definitions.
See [docs/teacher-checklist.md](docs/teacher-checklist.md) for the commands and
points to present in the next meeting.
See [docs/final-results.md](docs/final-results.md) for final reportable metrics.

## Gemini answer generation

Set the key directly in the terminal environment. Do not put it in source code,
JSONL, Git, or command-line arguments.

Generate validation prompts:

```bash
python scripts/export_qa_prompts.py \
	--questions data/processed/validation_retrieval.parquet \
	--limit 100 \
	--output results/qa_prompts_validation_100.jsonl
```

Generate answers with Gemini 3.5 Flash-Lite:

```bash
python scripts/generate_gemini_answers.py \
	--input results/qa_prompts_validation_100.jsonl \
	--output results/qa_predictions_validation_100.jsonl \
	--model gemini-3.5-flash-lite \
	--retry-errors
```

Evaluate answer F1 and citations:

```bash
python scripts/evaluate_answers.py \
	--predictions results/qa_predictions_validation_100.jsonl \
	--references data/processed/validation_retrieval.parquet \
	--output results/answer_metrics_validation_100.csv
```

The generator resumes successful records and logs failed records without
printing the API key.

When generated answers are available as JSONL, evaluate them with:

```bash
python scripts/evaluate_answers.py --predictions results/qa_predictions.jsonl
```

Export grounded prompts for an external or future LLM:

```bash
python scripts/export_qa_prompts.py --limit 100
```

Generate answers with Gemini. Set `GEMINI_API_KEY` directly in the terminal;
never put it in a file or command-line argument:

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
