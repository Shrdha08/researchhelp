## Retrieval results (dev split, 54 answerable questions)

| Config | Recall@5 | Recall@10 | MRR | PageRecall@5 | Median latency (s) |
|---|--:|--:|--:|--:|--:|
| semantic | 0.407 | 0.491 | 0.326 | 0.565 | 0.04 |
| hybrid | 0.602 | 0.722 | 0.473 | 0.722 | 0.03 |
| hybrid_rerank[ms-marco-MiniLM-L-6-v2, k_cand=10] | 0.667 | 0.750 | 0.559 | 0.769 | 0.17 |
| hybrid_rerank[ms-marco-MiniLM-L-6-v2, k_cand=20] | 0.639 | 0.769 | 0.560 | 0.759 | 0.31 |
| hybrid_rerank[ms-marco-MiniLM-L-6-v2, k_cand=30] | 0.657 | 0.769 | 0.564 | 0.778 | 0.50 |
| hybrid_rerank[jina-reranker-v1-turbo-en, k_cand=10] | 0.648 | 0.750 | 0.548 | 0.778 | 0.21 |
| hybrid_rerank[jina-reranker-v1-turbo-en, k_cand=20] | 0.611 | 0.722 | 0.541 | 0.704 | 0.40 |
| hybrid_rerank[jina-reranker-v1-turbo-en, k_cand=30] | 0.630 | 0.704 | 0.546 | 0.713 | 0.64 |
| hybrid_rerank[bge-reranker-base, k_cand=10] | 0.605 | 0.759 | 0.456 | 0.753 | 13.32 |

Recall@5 by question type:

| Config | comparison (n=9) | conceptual (n=11) | factual (n=34) |
|---|--:|--:|--:|
| semantic | 0.167 | 0.455 | 0.456 |
| hybrid | 0.222 | 0.727 | 0.662 |
| hybrid_rerank[ms-marco-MiniLM-L-6-v2, k_cand=10] | 0.333 | 0.727 | 0.735 |
| hybrid_rerank[ms-marco-MiniLM-L-6-v2, k_cand=20] | 0.278 | 0.727 | 0.706 |
| hybrid_rerank[ms-marco-MiniLM-L-6-v2, k_cand=30] | 0.278 | 0.727 | 0.735 |
| hybrid_rerank[jina-reranker-v1-turbo-en, k_cand=10] | 0.333 | 0.636 | 0.735 |
| hybrid_rerank[jina-reranker-v1-turbo-en, k_cand=20] | 0.333 | 0.636 | 0.676 |
| hybrid_rerank[jina-reranker-v1-turbo-en, k_cand=30] | 0.333 | 0.636 | 0.706 |
| hybrid_rerank[bge-reranker-base, k_cand=10] | 0.296 | 0.545 | 0.706 |
