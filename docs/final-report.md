# Thai Legal Question Answering and Retrieval System

## 1. Introduction

This project develops a Thai legal question answering system with a focus on improving legal document retrieval. The system retrieves relevant legal sections for a given Thai legal question and provides evidence that can be used by a grounded language model to generate an answer.

The main research focus is retrieval performance. In particular, the project compares lexical retrieval, semantic retrieval, hybrid retrieval, and a reranking-based pipeline.

## 2. Dataset

The project uses the WangchanX-Legal-ThaiCCL-RAG dataset, a Thai legal question answering and retrieval dataset covering corporate and commercial law.

The dataset contains questions, positive contexts, hard-negative contexts, positive answers, and hard-negative answers.

The local corpus contains 4,545 unique legal sections from 35 laws.

The local dataset was divided into:

- Training: 8,211 questions
- Validation: 1,643 questions
- Test: 3,742 questions

The official test set was kept untouched during model and configuration selection.

## 3. Retrieval Methods

Three main retrieval approaches were evaluated.

### 3.1 BM25

BM25 was used as a lexical retrieval baseline. It is effective when important legal terms in the question overlap with terms in the relevant legal section.

### 3.2 Semantic Retrieval

A MiniLM-based sentence embedding model was used to represent questions and legal sections as dense vectors. Semantic similarity was then used to retrieve relevant legal sections.

### 3.3 Hybrid Retrieval

BM25 and semantic retrieval were combined using a weighted hybrid score.

The validation set was used to select the weighting parameter. The selected configuration was alpha = 0.5.

## 4. Reranking Pipeline

The final QA retrieval pipeline uses two stages:

1. Hybrid retrieval generates the top-20 candidate legal sections.
2. A Cross-Encoder Reranker scores the question and candidate section together.
3. The top-5 reranked sections are provided as evidence to the answer generation stage.

The complete pipeline is:

Question → Hybrid Retrieval → Top-20 Candidates → Cross-Encoder Reranker → Top-5 Evidence → Grounded LLM Answer

This two-stage design allows the first stage to efficiently retrieve a broad candidate set while the reranker focuses on identifying the most relevant legal sections.

## 5. Retrieval Evaluation

Retrieval performance was evaluated using Recall@1, Recall@3, Recall@5, and Mean Reciprocal Rank (MRR).

| Method | Split | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---:|---:|---:|---:|
| BM25 | Test | 55.59% | 70.12% | 75.28% | 0.6328 |
| Semantic MiniLM | Test | 33.54% | 48.61% | 55.13% | 0.4168 |
| Hybrid alpha=0.5 | Test | 58.82% | 74.35% | 79.90% | 0.6701 |
| Hybrid MiniLM + Cross-Encoder Reranker (FINAL PIPELINE) | Test | 71.94% | 83.54% | 86.26% | 0.7790 |
| BGE-M3 | Validation | 65.43% | 81.13% | 86.73% | 0.7379 |

The BGE-M3 row is a validation-set comparison only. No verified BGE-M3 test-set result is reported here.

The final pipeline is Hybrid MiniLM with alpha = 0.5 followed by a Cross-Encoder Reranker. Its test-set result is reported in the table above.

## 6. Sampled Retrieval Verification

A 50-question sample from the validation set was manually inspected at the level of retrieved evidence and gold positive contexts.

The final top-5 evidence achieved:

- Recall@1: 68.0%
- Recall@3: 80.0%
- Recall@5: 82.0%
- MRR: 0.7407

One question had an empty `positive_contexts` field despite having a positive answer. This case was treated as a dataset issue rather than an ordinary retrieval error.

After excluding this case, 41 of 49 valid questions contained the gold context within the top five results, corresponding to 83.7%.

This sampled verification should not be interpreted as a replacement for the full benchmark evaluation.

## 7. Error Analysis

The sampled errors were primarily caused by incorrect ranking of legally related sections.

Common error patterns included:

- Retrieving the correct law but the wrong section.
- Retrieving sections with similar legal terminology.
- Retrieving a closely related law instead of the target law.
- Missing the exact gold section even when related sections were retrieved.

These observations suggest that legal retrieval errors are often fine-grained ranking errors rather than complete failures to identify the relevant legal domain.

This motivates the use of a Cross-Encoder Reranker as a second-stage ranking model.

## 8. Grounded Question Answering

The retrieved top-5 legal sections are inserted into a grounded prompt.

The answer generation prompt instructs the language model to:

- Answer using only the retrieved legal evidence.
- Avoid inventing legal provisions or details.
- Explicitly state when the retrieved evidence is insufficient.
- Identify the relevant law and section.

The system therefore separates retrieval from answer generation and provides explicit evidence to the generation model.

## 9. Answer Generation Evaluation

An answer-generation pipeline using the Gemini API was implemented together with an answer evaluator.

The evaluator supports:

- Token-level F1 against the reference answer.
- Citation correctness.
- Abstention detection.

However, full answer-level evaluation could not be completed because the available free API quota was exhausted during the experiment. Therefore, the partial generated responses are not used as a quantitative measure of final answer quality.

The retrieval evaluation is consequently the primary quantitative evaluation in this project.

## 10. Limitations

Several limitations remain.

First, the answer-generation stage was not evaluated on the complete evaluation set because of API quota limitations.

Second, human evaluation was limited to sampled retrieval verification rather than a full-scale expert evaluation.

Third, the current experiments focus primarily on retrieval quality. More detailed evaluation of citation correctness, answer faithfulness, abstention behavior, latency, and memory usage would strengthen the system evaluation.

Finally, the dataset itself contains at least one example where a positive answer exists but the corresponding positive context field is empty, demonstrating the need to account for dataset quality during evaluation.

## 11. Conclusion

This project implemented and evaluated a Thai legal retrieval and question answering pipeline.

The experiments compared BM25, semantic retrieval, hybrid retrieval, BGE-M3, and Cross-Encoder reranking. BGE-M3 was evaluated as a validation comparison and achieved Recall@1 of 65.43%, Recall@3 of 81.13%, Recall@5 of 86.73%, and MRR of 0.7379. No BGE-M3 test-set result is claimed.

The FINAL PIPELINE is Hybrid MiniLM with alpha = 0.5 and a Cross-Encoder Reranker. On the untouched test set it achieved Recall@1 of 71.94%, Recall@3 of 83.54%, Recall@5 of 86.26%, and MRR of 0.7790. It generates candidates and selects the final evidence for grounded answer generation.

Error analysis indicates that many retrieval failures occur between legally similar sections, supporting the use of reranking to improve fine-grained relevance ranking.

The project also provides a CLI demonstration, web demonstration, grounded prompting pipeline, answer evaluation tools, reproducibility documentation, and retrieval error analysis.
