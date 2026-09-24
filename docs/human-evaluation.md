# Human Evaluation Protocol

Automatic retrieval metrics do not establish that a generated Thai legal answer
is useful or safe. Use this protocol after generating answers from the grounded
prompts.

## Sampling

- Sample 50-100 questions from the untouched test set after the final system is
  locked.
- Do not use these judgments to change alpha, top-k, model or reranker.
- Have at least two reviewers score each answer when possible.

## Rubric

Score each item from 0 to 2:

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| Answer correctness | Wrong or misleading | Partly correct | Correct |
| Evidence faithfulness | Unsupported by context | Partly supported | Fully supported |
| Citation correctness | Wrong/missing | Partly correct | Correct law and section |
| Completeness | Misses the main issue | Partly addresses it | Addresses the question |
| Abstention behavior | Hallucinates when evidence is weak | Unclear | Correctly refuses or qualifies |

Recommended annotation columns:

```text
question
answer
retrieved_sources
answer_correctness
evidence_faithfulness
citation_correctness
completeness
abstention_behavior
notes
reviewer_id
```

Report the mean score per criterion, agreement between reviewers, and examples
of both successful and failed answers.