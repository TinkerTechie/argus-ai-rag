# RAG Evaluation Results

This document summarizes the performance of the Argus AI RAG system using the RAGAS framework.

## Dataset Overview
- **Total Questions:** 25
- **Categories:** Easy (Definitions), Medium (Workflows), Complex (Architectural Reasoning)
- **Model under test:** Phi3 (Local)
- **Embeddings:** all-MiniLM-L6-v2

## Metrics Summary

| Metric | With Critic Loop | Without Critic Loop |
| :--- | :---: | :---: |
| **Faithfulness** | 0.88 | 0.76 |
| **Answer Relevancy** | 0.91 | 0.82 |
| **Context Recall** | 0.85 | 0.85 |

## Interpretation of Results
1. **Faithfulness Improvement:** The Critic loop significantly increased faithfulness (+12%). By identifying hallucinations or unsupported claims, the Critic forces the Analyst to regenerate answers based strictly on the provided context.
2. **Answer Relevancy:** Relevancy improved (+9%) as the Critic ensures the Analyst directly addresses all parts of the user query.
3. **Context Recall Parity:** Recall remained identical (0.85) for both because the retrieval step occurs before the Critic loop.
4. **Conclusion:** The iterative multi-agent approach (LangGraph) provides a measurable boost in answer quality at the cost of additional latency per query.

---
*Results generated via `python -m scripts.evaluate_rag`*
