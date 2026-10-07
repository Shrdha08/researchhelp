# ResearchHelp

Upload research papers (PDF), select one or more, and ask questions:

- **Evidence Q&A**: grounded answers with paper and page citations ("What datasets does this paper use?", "Compare the methodologies of A and B").
- **Research assistance**: limitations, research gaps and proposed directions, in three separated sections: *evidence* (cited), *analysis* (inferred) and *proposed directions* (hypotheses, each with an experiment to test it).

A LangGraph router decides which of the two pipelines answers each question; `--mode` overrides it.

Stack: Python · LangChain · LangGraph · Qdrant · FastEmbed (bge-small, BM25, bge-reranker) · Groq · FastAPI · PostgreSQL · React · Docker.

See [docs/architecture.md](docs/architecture.md), [docs/decisions.md](docs/decisions.md) and [docs/evaluation.md](docs/evaluation.md).

## Results (test split, hand-labelled benchmark over 8 papers)

| Version | Recall@5 | MRR | Faithfulness | Correctness |
|---|--:|--:|--:|--:|
| Semantic search | 0.483 | 0.304 | 0.925 | 0.724 |
| Hybrid (dense + BM25, RRF) | 0.569 | 0.366 | 0.886 | 0.707 |
| Hybrid + cross-encoder reranker | **0.655** | **0.448** | **0.946** | **0.793** |

There are 29 answerable test questions; all 6 unanswerable questions were correctly refused by every version. Generation metrics come from an LLM judge (Qwen3.8-27B, a different model family from the gpt-oss-120b generator). Methodology, the reranker selection on the dev split, and limitations are in [docs/evaluation.md](docs/evaluation.md).

**Router and research assistant:** the LLM router (gpt-oss-20b) sent 52/52 labelled queries to the right pipeline, against 0.83 for a keyword baseline. Across 10 research prompts, every answer had all three sections and cited every selected paper. 89.5% of evidence claims were judged supported by the excerpts they cite, and the remaining failures are documented.

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Repository + architecture | done |
| 1 | Ingestion + baseline RAG (CLI) | done |
| 2 | Hybrid retrieval, reranking, evaluation | done (manual judge check pending) |
| 3 | Research assistant + LangGraph routing | done |
| 4 | FastAPI backend | done |
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

## API

```bash
cd backend
uv run researchhelp serve            # http://127.0.0.1:8000, Swagger UI at /docs
curl -F "files=@../data/papers/dpr.pdf" localhost:8000/papers/upload    # 202, status processing
curl localhost:8000/papers                                               # poll until "ready"
curl -H "Content-Type: application/json" localhost:8000/query      -d '{"question": "Which datasets are used?", "paper_ids": ["3e67fc1a9977715a"], "mode": "auto"}'
```

Endpoints: `POST /papers/upload`, `GET /papers`, `GET /papers/{id}`, `DELETE /papers/{id}`, `POST /query`, `GET /health`. Papers indexed with the CLI appear in the API automatically after a restart.

## Evaluation

```bash
cd ..                                                  # repo root
uv run --project backend python -m evaluation.validate_dataset
uv run --project backend python -m evaluation.retrieval_eval --split test
uv run --project backend python -m evaluation.generation_eval --split test --strategies semantic hybrid hybrid_rerank
```

## Usage (CLI)

```bash
cd backend
uv run python ../scripts/download_papers.py          # evaluation corpus -> data/papers/
uv run researchhelp inspect ../data/papers/dpr.pdf --page 5   # check parsing/chunking, no indexing
uv run researchhelp ingest ../data/papers/            # parse, embed, index
uv run researchhelp papers                            # list indexed papers
uv run researchhelp ask "What batch size and learning rate were used?" -p "dense passage"
uv run researchhelp ask "Compare the datasets used." -p "dense passage" -p "retrieval-augmented generation" --show-context
uv run researchhelp ask "What research gaps exist across these papers?" -p "dense passage" -p "retrieval-augmented generation" -p "leveraging passage"
uv run researchhelp ask "How could the retriever be improved?" -p "dense passage" --mode evidence   # force a pipeline
```

Every answer starts with the route taken, e.g. `[route: research (llm): ...]`. The route source is `llm`, `keyword` (fallback) or `forced` (from `--mode`).

`-p` takes a paper-ID prefix or a fragment of the title. Answers cite `[S#]` markers, which are resolved to paper, page and section from chunk metadata. Markers the model invents are dropped.
