## Retrieval results (test split, 29 answerable questions)

| Config | Recall@5 | Recall@10 | MRR | PageRecall@5 | Median latency (s) |
|---|--:|--:|--:|--:|--:|
| semantic | 0.483 | 0.586 | 0.304 | 0.655 | 0.03 |
| hybrid | 0.569 | 0.741 | 0.366 | 0.759 | 0.03 |
| hybrid_rerank[ms-marco-MiniLM-L-6-v2, k_cand=10] | 0.655 | 0.741 | 0.448 | 0.724 | 0.10 |

Recall@5 by question type:

| Config | comparison (n=4) | conceptual (n=4) | factual (n=21) |
|---|--:|--:|--:|
| semantic | 0.250 | 1.000 | 0.429 |
| hybrid | 0.375 | 1.000 | 0.524 |
| hybrid_rerank[ms-marco-MiniLM-L-6-v2, k_cand=10] | 0.250 | 0.750 | 0.714 |
