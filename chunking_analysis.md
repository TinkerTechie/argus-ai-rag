# Chunking Analysis Report

Evaluating the optimal chunk size for retrieval performance in Argus AI.

## Experiment Setup
- **Chunk Sizes Tested:** 256, 500, 1000
- **Overlap:** 100 characters (fixed)
- **Metric:** Context Recall (Recall@5)
- **Embedding Model:** all-MiniLM-L6-v2

## Results

| Chunk Size | Recall@5 | avg. Retrieval Latency |
| :--- | :---: | :---: |
| 256 | 0.72 | 12ms |
| **500** | **0.85** | **18ms** |
| 1000 | 0.81 | 25ms |

## Analysis & Final Decision
- **Small Chunks (256):** Suffered from context fragmentation. Relevant information was often split across multiple chunks, making it harder for the embedding model to capture the full semantic context.
- **Large Chunks (1000):** Introduced too much noise. Large chunks contained unrelated information, which diluted the similarity score and allowed irrelevant context to rank higher.
- **Chosen Size (500):** **500 characters** provided the best balance. It maintained high retrieval precision (0.85) while keeping latency low enough for local inference.

**Decision:** Default chunk size is set to **500 characters** with a **100-character overlap**.

---
*Results generated via `python -m scripts.chunking_experiment`*
