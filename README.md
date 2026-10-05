# ResearchHelp

Upload research papers (PDF), select one or more, and ask questions:

- **Evidence Q&A**: grounded answers with paper and page citations ("What datasets does this paper use?", "Compare the methodologies of A and B").
- **Research assistance** *(Phase 3)*: limitations, research gaps and proposed directions, with evidence kept separate from inference.

Stack: Python · LangChain · LangGraph · Qdrant · FastEmbed (bge-small, BM25, bge-reranker) · Groq · FastAPI · PostgreSQL · React · Docker.

See [docs/architecture.md](docs/architecture.md), [docs/decisions.md](docs/decisions.md) and [docs/evaluation.md](docs/evaluation.md).

## Results (test split, hand-labelled benchmark over 8 papers)

| Version | Recall@5 | MRR | Faithfulness | Correctness |
|---|--:|--:|--:|--:|
| Semantic search | 0.483 | 0.304 | 0.925 | 0.724 |
| Hybrid (dense + BM25, RRF) | 0.569 | 0.366 | 0.886 | 0.707 |
| Hybrid + cross-encoder reranker | **0.655** | **0.448** | **0.946** | **0.793** |

There are 29 answerable test questions; all 6 unanswerable questions were correctly refused by every version. Generation metrics come from an LLM judge (Qwen3.8-27B, a different model family from the gpt-oss-120b generator). Methodology, the reranker selection on the dev split, and limitations are in [docs/evaluation.md](docs/evaluation.md).

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Repository + architecture | done |
| 1 | Ingestion + baseline RAG (CLI) | done |
| 2 | Hybrid retrieval, reranking, evaluation | done (manual judge check pending) |
| 3 | Research assistant + LangGraph routing | planned |
| 4 | FastAPI backend | planned |
| 5 | PostgreSQL persistence | planned |
| 6 | React frontend | planned |
| 7 | Tests, Docker, deployment | planned |

## Local setup

Requirements: [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for the project) and Docker.

```bash
cp .env.example .env            # add your GROQ_API_KEY
docker compose up -d qdrant postgres
cd backend
uv sync --extra gpu             # NVIDIA GPU (CUDA 13 driver); or: uv sync --extra cpu
uv run pytest
```

One of the two extras is required, because the ONNX runtime for the local embedding and reranker models comes from it. `ONNX_DEVICE=auto` (the default) uses the GPU when available and logs which provider is active.

To run without Docker, set `QDRANT_PATH=./data/qdrant_local` in `.env` to use embedded on-disk Qdrant.

## Evaluation

```bash
cd ..                                                  # repo root
uv run --project backend python -m evaluation.validate_dataset
uv run --project backend python -m evaluation.retrieval_eval --split test
uv run --project backend python -m evaluation.generation_eval --split test --strategies semantic hybrid hybrid_rerank
```

## Usage (CLI, Phase 1)

```bash
cd backend
uv run python ../scripts/download_papers.py          # evaluation corpus -> data/papers/
uv run researchhelp inspect ../data/papers/dpr.pdf --page 5   # check parsing/chunking, no indexing
uv run researchhelp ingest ../data/papers/            # parse, embed, index
uv run researchhelp papers                            # list indexed papers
uv run researchhelp ask "What batch size and learning rate were used?" -p "dense passage"
uv run researchhelp ask "Compare the datasets used." -p "dense passage" -p "retrieval-augmented generation" --show-context
```

`-p` takes a paper-ID prefix or a fragment of the title. Answers cite `[S#]` markers, which are resolved to paper, page and section from chunk metadata. Markers the model invents are dropped.
