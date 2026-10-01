# Final Results

## Retrieval and reranking

```text
Hybrid MiniLM -> alpha=0.5 -> candidate top-20
-> Cross-Encoder reranker -> evidence top-5
```

```text
Recall@1: 0.7194
Recall@3: 0.8354
Recall@5: 0.8626
MRR@5:    0.7790
```

## Gemini answer generation

```text
Model: gemini-3.5-flash-lite
Validation questions: 100
Test questions: 100
```

Validation:

```text
Answer Token F1:  0.395647
Citation Correct: 0.640000
Abstained:        0.280000
```

Untouched test sample:

```text
Answer Token F1:  0.487709
Citation Correct: 0.800000
Abstained:        0.130000
```

Generated prediction JSONL files are local artifacts and are ignored by Git.

## Limitations

- Answer evaluation uses 100-question validation and test samples.
- Category labels are rule-based, not gold annotations.
- The environment has no NVIDIA CUDA GPU.
- Some informal questions still retrieve weak evidence.