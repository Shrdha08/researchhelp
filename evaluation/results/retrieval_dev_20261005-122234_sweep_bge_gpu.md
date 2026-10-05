## Retrieval results (dev split, 54 answerable questions)

| Config | Recall@5 | Recall@10 | MRR | PageRecall@5 | Median latency (s) |
|---|--:|--:|--:|--:|--:|
| hybrid_rerank[bge-reranker-base, k_cand=10] | 0.605 | 0.759 | 0.453 | 0.753 | 2.19 |
| hybrid_rerank[bge-reranker-base, k_cand=20] | 0.614 | 0.722 | 0.464 | 0.781 | 12.48 |
| hybrid_rerank[bge-reranker-base, k_cand=30] | 0.596 | 0.657 | 0.442 | 0.753 | 20.09 |

Recall@5 by question type:

| Config | comparison (n=9) | conceptual (n=11) | factual (n=34) |
|---|--:|--:|--:|
| hybrid_rerank[bge-reranker-base, k_cand=10] | 0.296 | 0.545 | 0.706 |
| hybrid_rerank[bge-reranker-base, k_cand=20] | 0.241 | 0.545 | 0.735 |
| hybrid_rerank[bge-reranker-base, k_cand=30] | 0.241 | 0.545 | 0.706 |
