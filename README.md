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
| 1 | Ingestion + baseline RAG (CLI) | in progress |
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
