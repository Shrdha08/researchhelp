# ResearchHelp

Upload research papers (PDF), select one or more, and ask questions:

- **Evidence Q&A**: grounded answers with paper and page citations ("What datasets does this paper use?", "Compare the methodologies of A and B").
- **Research assistance** *(Phase 3)*: limitations, research gaps and proposed directions, with evidence kept separate from inference.

Stack: Python · LangChain · LangGraph · Qdrant · FastEmbed (bge-small, BM25, bge-reranker) · Groq · FastAPI · PostgreSQL · React · Docker.

See [docs/architecture.md](docs/architecture.md) and [docs/decisions.md](docs/decisions.md).

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Repository + architecture | done |
| 1 | Ingestion + baseline RAG (CLI) | done |
| 2 | Hybrid retrieval, reranking, evaluation | planned |
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
uv sync
uv run pytest
```

To run without Docker, set `QDRANT_PATH=./data/qdrant_local` in `.env` to use embedded on-disk Qdrant.

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
