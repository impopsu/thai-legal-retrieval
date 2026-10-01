# Context Metrics

The context evaluation compares retrieved document `unique_key` values with
the `positive_contexts` in the dataset.

Metrics:

- `ContextRecall@k`: whether at least one positive context appears in top-k.
- `ContextPrecision@5`: the fraction of top-five results that are positive.
- `ContextMRR@5`: reciprocal rank of the first positive context within top-five.

Run on a validation sample:

```bash
python scripts/evaluate_context.py \
  --questions data/processed/validation_retrieval.parquet \
  --documents data/processed/legal_documents_categorized.csv \
  --limit 100
```

This is separate from answer evaluation. Answer F1 compares generated LLM
answers with `positive_answer` and is run with `scripts/evaluate_answers.py`.