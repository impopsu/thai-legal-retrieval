# Thai Legal QA

Thai legal retrieval and grounded question answering using BM25, multilingual
semantic search, hybrid retrieval and reranking.

## Current status

The reusable core is in `legal_qa/`. Original experiments remain in `experiments/`
for comparison. The selected final pipeline is Hybrid MiniLM with alpha `0.5`,
candidate top-20 and Cross-Encoder reranking to top-5. The hybrid score
convention is:

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

See [docs/project-summary.md](docs/project-summary.md) for dataset, methods,
metrics, final results, limitations and presentation notes.

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
