# Thai Legal Question Answering and Retrieval System

## Slide 1 — Title

**Thai Legal Question Answering and Retrieval System**

Thai Legal QA / RAG

Computer Engineering  
Kasetsart University

---

## Slide 2 — Problem

### Problem

Thai legal questions often require finding the exact legal section before generating an answer.

Challenges:

- Legal sections contain similar terminology.
- The correct law may be identified while the exact section is missed.
- Pure semantic or lexical retrieval can return closely related but incorrect sections.
- LLM answers need reliable legal evidence to reduce hallucination.

### Goal

Improve legal evidence retrieval and provide grounded evidence for answer generation.

---

## Slide 3 — Dataset

### WangchanX-Legal-ThaiCCL-RAG

Thai legal QA/RAG dataset focused on corporate and commercial law.

**Local data**

- 35 laws
- 4,545 unique legal sections
- 8,211 training questions
- 1,643 validation questions
- 3,742 test questions

Each QA example contains:

- Question
- Positive context
- Hard-negative context
- Positive answer
- Hard-negative answer

---

## Slide 4 — System Architecture

### Proposed Pipeline

**Question**

↓

**Hybrid Retrieval**

BM25 + Semantic Retrieval

↓

**Top-20 Candidates**

↓

**Cross-Encoder Reranker**

↓

**Top-5 Evidence**

↓

**Grounded LLM**

↓

**Answer + Citation**

The first stage retrieves candidates efficiently, while the reranker performs fine-grained relevance ranking.

---

## Slide 5 — Retrieval Methods

### Methods Compared

**BM25**

Lexical retrieval based on term matching.

**MiniLM**

Dense semantic retrieval using sentence embeddings.

**Hybrid**

Combines BM25 and semantic retrieval.

**BGE-M3**

Multilingual embedding model evaluated as a stronger semantic retrieval baseline.

---

## Slide 6 — Validation: Choosing Hybrid Weight

### Hybrid Alpha

The validation set was used to select the weighting parameter.

The best validation MRR was obtained at:

**alpha = 0.5**

This configuration was used for the final Hybrid retrieval pipeline.

---

## Slide 7 — Test Results

### Retrieval Performance

| Method | Split | R@1 | R@3 | R@5 | MRR |
|---|---|---:|---:|---:|---:|
| BM25 | Test | 55.59% | 70.12% | 75.28% | 0.6328 |
| Semantic MiniLM | Test | 33.54% | 48.61% | 55.13% | 0.4168 |
| Hybrid α=0.5 | Test | 58.82% | 74.35% | 79.90% | 0.6701 |
| Hybrid MiniLM + Cross-Encoder Reranker (FINAL) | Test | 71.94% | 83.54% | 86.26% | 0.7790 |
| BGE-M3 | Validation | 65.43% | 81.13% | 86.73% | 0.7379 |

**Observation**

The final pipeline is Hybrid MiniLM + Cross-Encoder Reranker. BGE-M3 is shown only as a validation comparison; no BGE-M3 test-set result is claimed.

---

## Slide 8 — Error Analysis

### Sampled Verification

50 validation questions were inspected.

Observed errors:

- Wrong section within the same law.
- Similar legal terminology.
- Related but incorrect law.
- Exact gold section ranked below related sections.
- One dataset example had no positive context.

### Main Finding

Many errors are **fine-grained ranking errors** rather than complete retrieval failures.

This supports the use of a second-stage Cross-Encoder Reranker.

---

## Slide 9 — Grounded QA

### Grounded Answer Generation

The top-5 retrieved legal sections are provided to the LLM as evidence.

The prompt instructs the model to:

- Use only retrieved evidence.
- Avoid inventing legal provisions.
- State when evidence is insufficient.
- Identify the relevant law and section.

This separates **retrieval** from **answer generation**.

---

## Slide 10 — Evaluation Limitation

### Answer-Level Evaluation

An answer-generation pipeline and evaluator were implemented.

Metrics:

- Token-level F1
- Citation correctness
- Abstention

However, full answer-level evaluation was not completed because the available free Gemini API quota was exhausted.

Therefore:

**Retrieval performance is the primary quantitative evaluation.**

---

## Slide 11 — Demo

### System Demo

Demonstrate:

1. Enter a Thai legal question.
2. Retrieve relevant legal sections.
3. Rerank the candidate evidence.
4. Generate a grounded answer.
5. Display the cited legal source.

---

## Slide 12 — Conclusion

### Conclusion

The project implemented a Thai legal QA/RAG pipeline with multi-stage retrieval.

Key results:

- BM25 provides a strong lexical baseline.
- Hybrid retrieval improves over BM25.
- Hybrid MiniLM + Cross-Encoder Reranker is the FINAL PIPELINE, with test-set R@1 71.94%, R@3 83.54%, R@5 86.26%, and MRR 0.7790.
- BGE-M3 is a validation comparison, not the final pipeline.
- Error analysis shows that exact-section ranking remains challenging.
- Cross-Encoder reranking provides a suitable second-stage approach.
- Grounded prompting connects retrieved legal evidence to LLM answer generation.

### Future Work

- Full expert human evaluation.
- More detailed citation evaluation.
- Latency and memory analysis.
- More reranker model comparisons.
- Full-scale answer generation and evaluation.
