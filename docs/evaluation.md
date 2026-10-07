# Evaluation

Every number in this document is produced by the committed scripts in `evaluation/`. The raw outputs (JSON with per-question scores, plus markdown tables) are in `evaluation/results/`.

## Corpus

Eight arXiv papers on retrieval-augmented generation and dense retrieval (`evaluation/datasets/papers.yaml`): RAG, DPR, REALM, FiD, ColBERT, RETRO, Self-RAG and HyDE. 837 chunks in total. The papers were chosen because they share datasets (Natural Questions, TriviaQA, MS MARCO) and methods, which makes multi-paper comparison questions meaningful.

## Dataset

`evaluation/datasets/questions.jsonl` has 90 hand-written questions:

| Type | n | Purpose |
|---|--:|---|
| factual | 55 | Specific facts: datasets, hyperparameters, numbers. Many depend on exact terms ("batch size of 128", "SCaNN", "13-gram Jaccard") |
| conceptual | 15 | Explanations of a method ("What is late interaction?") |
| comparison | 13 | Two or three papers; the evidence comes from each paper |
| unanswerable | 7 | The papers don't contain the answer; tests refusing to answer |

**Gold labels** for each answerable question are evidence items `(paper, page, quote)`, where the quote is a short exact span from the paper. A retrieved chunk is *relevant* to an evidence item when it comes from the same paper and page **and** contains the quote (case- and whitespace-insensitive).

Why quotes and not chunk IDs: labels must survive changes to chunking (chunk size, overlap, cleaning). With chunk-ID labels, every chunking experiment would need relabelling.

`python -m evaluation.validate_dataset` checks that all 111 quotes appear on their labelled pages as our own parser extracts them, and that each lies fully inside at least one chunk. A quote split across two chunks couldn't be matched by any retriever.

**Split.** Each question is assigned to dev or test (about 70/30) by hashing its ID. Hyperparameters such as reranker choice and `k_candidates` are chosen on **dev**; headline numbers are reported on **test**. This stops the configuration being tuned to the very questions it's scored on.

**Limitations of the dataset.** The questions were written by the project author, with an AI assistant's help in finding passages, from text the system parses. A reader reviewing the questions independently is the next step. The set is small (35 test questions), so differences of a few points are within noise.

## Retrieval metrics (no LLM involved)

The retriever returns its top 10 chunks for the question's own papers.

- **Recall@K**: for each question, the fraction of its gold evidence items found in the top K chunks, averaged over questions. A 3-paper comparison that only finds two papers' evidence scores 0.67, so this directly measures multi-paper coverage.
- **MRR**: the mean of 1 / (rank of the first relevant chunk), with 0 if none appears in the top 10. It rewards putting evidence first.
- **PageRecall@5**: a looser variant that only requires the right paper and page.

## Generation metrics (LLM judge)

The generator is `openai/gpt-oss-120b` and the judge is `qwen/qwen3.8-27b`, both on Groq. They come from different model families, which reduces the judge favouring answers from its own model family. The metrics are implemented in `evaluation/generation_eval.py`, with the prompts visible in the file, rather than through RAGAS. The definitions follow RAGAS:

- **Faithfulness**: the judge splits the answer into atomic claims and marks each as supported or not by the retrieved context. Score = supported / total. It measures hallucination relative to the context, not whether the answer is correct.
- **Answer relevance**: the judge writes three questions the answer would answer. Score = their mean cosine similarity (bge-small) to the real question. It measures whether the answer addresses the question.
- **Context relevance**: the judge labels each retrieved chunk as relevant or not. Score = relevant / retrieved.
- **Correctness**: the judge compares the answer with the human reference answer (correct = 1, partial = 0.5, incorrect = 0).
- **Refusal accuracy**: correctness on the unanswerable questions, where "correct" means the system said the papers don't provide the answer.

**Judge validity.** `generation_eval.py` writes `manual_review_*.md`, a random sample of 15 answers to grade by hand. Agreement between those human grades and the judge is reported below; without that check, the judge's numbers should be treated as indicative only.

Groq's free tier allows 8,000 tokens per minute per model, so every generation and judge output is cached in `evaluation/results/generation_cache/` and runs can resume.

## Results

All runs: 8 papers, 834 chunks, `k_final = 6` chunks sent to the LLM, `k_per_paper = 8`. GPU means a GTX 1650 (4 GB) through the ONNX CUDA provider; CPU means the same code with `ONNX_DEVICE=cpu`. Retrieval metrics are identical on both devices; only latency differs.

### 1. Choosing the reranker (dev split, 54 answerable questions)

Source: `retrieval_dev_20261005-114635_sweep_gpu.md` and `retrieval_dev_20261005-122234_sweep_bge_gpu.md`.

| Config | Recall@5 | Recall@10 | MRR |
|---|--:|--:|--:|
| semantic | 0.407 | 0.491 | 0.326 |
| hybrid | 0.602 | 0.722 | 0.473 |
| hybrid + MiniLM-L-6, k_cand=10 | **0.667** | 0.750 | 0.559 |
| hybrid + MiniLM-L-6, k_cand=20 | 0.639 | 0.769 | 0.560 |
| hybrid + MiniLM-L-6, k_cand=30 | 0.657 | 0.769 | **0.564** |
| hybrid + Jina-turbo, k_cand=10 | 0.648 | 0.750 | 0.548 |
| hybrid + Jina-turbo, k_cand=20 | 0.611 | 0.722 | 0.541 |
| hybrid + Jina-turbo, k_cand=30 | 0.630 | 0.704 | 0.546 |
| hybrid + bge-reranker-base, k_cand=10 | 0.605 | 0.759 | 0.453 |
| hybrid + bge-reranker-base, k_cand=20 | 0.614 | 0.722 | 0.464 |
| hybrid + bge-reranker-base, k_cand=30 | 0.596 | 0.657 | 0.442 |

**Decision: MiniLM-L-6 (`Xenova/ms-marco-MiniLM-L-6-v2`) with `k_candidates = 10`.** It had the best Recall@5 (the top 5 is roughly what the LLM sees) and the lowest latency. A larger candidate pool didn't help any reranker. The largest model, bge-reranker-base, was the slowest and on this dataset no better than hybrid search alone. Its latency in the pipeline ranged from 0.7 to 20 s per query on the 4 GB GPU, depending on memory pressure from other processes, versus 0.17–0.5 s for MiniLM.

### 2. Retrieval: headline numbers (test split, 29 answerable questions)

Source: `retrieval_test_20261005-125808_headline.md`, `..._133237_headline_bge.md` and `..._133308_cpu.md`.

| Version | Recall@5 | Recall@10 | MRR | PageRecall@5 | Latency GPU | Latency CPU |
|---|--:|--:|--:|--:|--:|--:|
| V1 semantic | 0.483 | 0.586 | 0.304 | 0.655 | 0.03 s | 0.16 s |
| V2 hybrid (dense + BM25, RRF) | 0.569 | 0.741 | 0.366 | 0.759 | 0.03 s | 0.17 s |
| **V3 hybrid + MiniLM reranker** | **0.655** | **0.741** | 0.448 | 0.724 | 0.10 s | 1.48 s |
| *reference: hybrid + bge-reranker-base* | 0.638 | 0.724 | **0.487** | n/a | 0.72 s | n/a |

Recall@5 by question type:

| Version | factual (n=21) | conceptual (n=4) | comparison (n=4) |
|---|--:|--:|--:|
| semantic | 0.429 | 1.000 | 0.250 |
| hybrid | 0.524 | 1.000 | 0.375 |
| hybrid + rerank | **0.714** | 0.750 | 0.250 |

From V1 to V3, Recall@5 rose from 0.483 to 0.655 (+0.17) and MRR from 0.304 to 0.448 (+0.14). Most of the gain is on factual questions, many of which hinge on exact terms ("batch size of 128", "SCaNN", "13-gram Jaccard"). That's where BM25 and then the cross-encoder help. On test, bge-reranker-base had a higher MRR than the chosen MiniLM (0.487 vs 0.448) but lower Recall@5. The choice was made on dev and isn't revised on test results, because that would turn the test split into a tuning set.

### 3. Generation (test split, 35 questions: 29 answerable and 6 unanswerable)

Source: `generation_test_20261005-134150.md`. Generator `openai/gpt-oss-120b` (reasoning effort low), judge `qwen/qwen3.8-27b`.

| Version | Faithfulness | Answer relevance | Context relevance | Correctness | Refusal accuracy |
|---|--:|--:|--:|--:|--:|
| V1 semantic | 0.925 | 0.884 | 0.454 | 0.724 | 6/6 |
| V2 hybrid | 0.886 | 0.887 | 0.517 | 0.707 | 6/6 |
| **V3 hybrid + rerank** | **0.946** | **0.888** | **0.546** | **0.793** | 6/6 |

- V3 is best on every metric. Context relevance rises steadily across V1 → V3 (0.454 → 0.546), matching the retrieval results.
- **V2 is not better than V1 on faithfulness or correctness, despite better retrieval.** Comparing the two per question, 7 of 29 answers differ in correctness, 3 in V2's favour and 4 against. Two of V2's losses were inspected by hand:
  - *hyde-06*: the right chunk was retrieved, but the generator read the MAP column of a flattened table instead of nDCG@10 (see Limitations).
  - *dpr-11*: V2 missed the gold passage, and the model then claimed the information wasn't there although another retrieved chunk implied it.

  At n = 29, differences of about ±0.04 are within noise, so the V1-vs-V2 generation difference isn't treated as meaningful.
- **Refusals:** all 6 unanswerable questions were correctly answered with "the papers don't provide this" in every version.
- **Answer relevance** barely moves (0.884–0.888). It measures whether the answer addresses the question, which the grounded prompt ensures regardless of retrieval quality.

### 4. Judge validity

`manual_review_20261005-134150.md` holds 15 randomly sampled answers across the three versions, with no version names or judge verdicts shown, for blind grading. The judge's verdicts are in `manual_review_20261005-134150_key.md`. **Agreement between the human grades and the judge: pending manual grading.** Until it's filled in, the generation numbers are indicative. In the two answers inspected so far (hyde-06, dpr-11), the judge's verdicts and claim-level reasons were correct.

## Limitations

- **Small test set.** With 29 answerable test questions, and only 4 each for conceptual and comparison, per-type numbers can swing by 0.25 from a single question.
- **Comparison questions remain the weakest** (Recall@5 0.25–0.375). The per-paper fan-out guarantees each paper is *represented*, but not that its most relevant passage is chosen.
- **Tables are extracted as flat text**, so numbers lose their column headers (hyde-06). Table-aware parsing is out of scope.
- **The questions were written by the project author**, with an AI assistant's help in locating passages. The answers were labelled from the same parsed text the system retrieves from.
- **The LLM judge** comes from a different model family than the generator, but it is still an LLM. The manual agreement check above addresses this.

---

# Phase 3: Router and research assistant

## 5. Router (intent classification)

Source: `results/router_20261007-231352.md`, from `python -m evaluation.router_eval`. There are 52 labelled queries in `datasets/router_queries.jsonl`: 28 evidence and 24 research. They include a deliberately hard subset where evidence and research questions are easy to confuse, such as "What limitations did the authors identify?" (evidence) versus "What are the limitations of these approaches?" (research). See ADR-13.

| Router | Accuracy | Accuracy on the 44 queries not used as few-shot examples |
|---|--:|--:|
| LLM router (gpt-oss-20b, few-shot JSON) | **1.000** (52/52) | **1.000** |
| Keyword router (the fallback, as a baseline) | 0.827 | 0.795 |

- The keyword router fails on paraphrased research requests with no trigger word: "What is missing from the evaluation…", "Which assumptions … might not hold", "Is there a common blind spot…". It also fires falsely on author-stated content that contains a trigger word: "What *improvements* over BM25 does DPR report?", "What *open problems* does the RAG paper say remain?".
- None of the 52 LLM calls needed the keyword fallback.
- **Caveat:** the queries, the labels and the router's few-shot prompt were all written by the project author, with an AI assistant's help. Eight queries appear verbatim as few-shot examples, which is why accuracy is also reported without them. A set of queries written by someone else would be the stronger test; 1.000 on 52 self-written queries shouldn't be read as "the router never fails".

## 6. Research assistant

Source: `results/research_20261007-233328_v3.md` and the cache `results/research_cache_v3/`, from `python -m evaluation.research_eval`. There are 10 research prompts over 1–3 papers each (`datasets/research_questions.jsonl`). Research answers have no ground truth, so the evaluation measures what can be checked: structure, grounding, and the faithfulness of the **evidence** section. Analysis and directions are inferences by design, so "faithfulness" doesn't apply to them; they are graded by hand with the rubric sheet `results/research_rubric_20261007-233328_v3.md`.

| Metric | v1 | v2 | **v3 (final)** |
|---|--:|--:|--:|
| Answers with all three sections | 0.9 (one `no_evidence`) | 1.0 | **1.0** |
| Answers citing every selected paper | 0.9 | 1.0 | **1.0** |
| Analysis/direction items linked to ≥1 evidence item | 1.0 | 1.0 | **1.0** |
| Items removed by validation (invented sources, ungrounded inference) | 0 | 0 | **0** |
| Mean items per answer (evidence / analysis / directions) | 5.9 / 4.2 / 3.0 | 7.9 / 5.1 / 3.7 | 8.6 / 4.7 / 3.5 |
| **Evidence faithfulness** (judge: claim supported by its cited excerpts) | 0.881 | 0.785 | **0.895** (77/86) |

**How v3 was reached.** These iterations were driven by the evaluation, and they are reported because they show what each prompt change did:

- **v1** left one question (adapting HyDE to legal documents) with no evidence. The extraction prompt said "return an empty list if nothing is relevant", and the model judged HyDE's own method description irrelevant to a *new* domain.
- The first judge also truncated each excerpt to 900 characters, which hid facts near the end of a chunk and produced false "unsupported" verdicts. It was replaced by a judge that shows every cited chunk once, in full, and explicitly counts added interpretation as unsupported. That judge is stricter: v1's answers scored 0.932 under the old judge and **0.881 under the new one**. All numbers in the table use the new judge.
- **v2** told the extraction step that a method's own description is relevant evidence even for new settings. Completeness rose to 1.0, but **faithfulness fell to 0.785** under the same judge. The model started putting *suggestions* into the evidence section ("Compare FiD performance when…"), including one factual error (FiD sizes given as 770M/11B; they are 220M/770M).
- **v3** kept that rule and added an explicit ban on suggestions, experiments and conclusions in the evidence step. The result is complete answers **and** the best faithfulness.

**Remaining failure pattern (9 of 86 claims).** A correct factual core with an interpretive tail, for example "…DPR outperforms BM25 except on SQuAD, *indicating a gap* in…" or "…evaluated on MS MARCO and TREC CAR, *without evaluation on* other QA datasets". These are mild versions of the leak v2 showed: the step that should only restate the papers adds a small inference.

**Caveats.** There are 10 questions, and v2 and v3 were tuned on these same 10, so there is no held-out set here. The faithfulness verdicts come from an LLM judge. The quality of the analysis and directions (specificity, grounding, plausibility) is only measured by the manual rubric, which is pending.
