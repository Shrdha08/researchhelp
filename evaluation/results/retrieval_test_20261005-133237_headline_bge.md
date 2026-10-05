## Retrieval results (test split, 29 answerable questions)

| Config | Recall@5 | Recall@10 | MRR | PageRecall@5 | Median latency (s) |
|---|--:|--:|--:|--:|--:|
| hybrid_rerank[bge-reranker-base, k_cand=10] | 0.638 | 0.724 | 0.487 | 0.776 | 0.72 |

Recall@5 by question type:

| Config | comparison (n=4) | conceptual (n=4) | factual (n=21) |
|---|--:|--:|--:|
| hybrid_rerank[bge-reranker-base, k_cand=10] | 0.125 | 1.000 | 0.667 |
