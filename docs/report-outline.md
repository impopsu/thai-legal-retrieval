# Report Outline

1. **Problem definition**: Thai legal question answering with grounded evidence.
2. **Dataset**: WangchanX-Legal-ThaiCCL-RAG, fields, split and limitations.
3. **Related work**: NitiBench and the free-format legal QA system.
4. **Methods**: BM25, MiniLM, BGE-M3, hybrid scoring and cross-encoder reranking.
5. **Evaluation protocol**: validation tuning and untouched test reporting.
6. **Results**: retrieval benchmark, reranker improvement, runtime and memory.
7. **Grounded QA pipeline**: evidence selection, prompt constraints and citation.
8. **Human evaluation**: correctness, faithfulness, citation and abstention rubric.
9. **Error analysis**: informal questions, weak evidence and corpus coverage.
10. **Limitations**: no LLM generator selected, CPU cost, and benchmark-derived corpus.
11. **Conclusion**: selected configuration and observed trade-offs.