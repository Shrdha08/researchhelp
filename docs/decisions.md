# Design decisions

Short architecture decision records (ADRs). The project should be able to defend each one.

## ADR-1: One scoped retriever, not single- vs multi-paper pipelines
Scope is a *filter*, not a *pipeline*. `ScopedRetriever` takes `paper_ids`, and a single paper is just a list of length one. For several papers it **fans out** (top-k per paper with `paper_id == X`), then merges. Without fan-out, a long or very on-topic paper can take every top-k slot, and a comparison answer would silently leave out the other papers.

## ADR-2: Hybrid search inside Qdrant (dense + sparse BM25, RRF fusion)
An in-memory BM25 retriever next to the vector store would need its own `paper_id` filtering and a rebuild on every upload, and it adds a second index to keep consistent. Qdrant stores a named dense vector and a named sparse vector per point and fuses them server-side with Reciprocal Rank Fusion, so a single filter applies to both signals. RRF fuses *ranks*, which avoids the problem that cosine and BM25 scores aren't on comparable scales. The sparse vector is written from Phase 1, so Phase 2 needs no re-ingestion.

## ADR-3: Chunks never cross page boundaries
Each page is split independently, so every chunk has exactly one page number, and citations become metadata lookups instead of guesses. The cost: a sentence that crosses a page break gets split. Overlap within a page is unaffected.

## ADR-4: The LLM cites source IDs, never page numbers
The context is formatted as `[S1] (paper, p.5, §Method) ...`. The model writes `[S1]`, and the code maps each ID back to paper, page and section. IDs that aren't in the context are dropped, so citations can't be fabricated.

## ADR-5: Research assistant = evidence-extraction chain + synthesis chain (Phase 3)
Chain A extracts cited evidence. Chain B sees only that evidence and produces analysis and proposed directions linked to evidence IDs. The separation between "what the papers say" and "what is inferred" is enforced by the structure, not only by prompt wording.

## ADR-6: LangGraph decides intent only (Phase 3)
Scope comes from explicit selection in the UI (`paper_ids`). The graph routes evidence questions and research questions, and the user can override it with `mode`.

## ADR-7: PostgreSQL for application state, Qdrant for vectors
Postgres holds papers, ingestion status and conversations: relational, transactional data. Qdrant holds chunks and vectors: filtered approximate nearest-neighbour search plus native sparse and dense hybrid search. `paper_id` = the first 16 hex characters of the PDF's SHA-256. It's deterministic, deduplicates uploads, and links the two stores. pgvector would be a valid alternative; Qdrant was chosen for its built-in hybrid search.

## ADR-8: FastAPI contains no RAG logic
Routes → services → graph/pipelines. The RAG core can be used and tested from the CLI.

## ADR-9: Local ONNX models via FastEmbed; Groq for generation
Embeddings (bge-small-en-v1.5), BM25 sparse vectors and the cross-encoder reranker run locally on CPU through FastEmbed. It uses the ONNX runtime without PyTorch, which keeps the Docker image small and costs nothing per call. Generation uses Groq-hosted open models on the free tier, and LLM calls are retried with backoff.

## ADR-10: qdrant-client directly instead of langchain-qdrant
Fan-out, flat payloads and the Query API (prefetch + RRF) are clearer written directly. The retriever is a plain class that returns LangChain `Document`s; chains compose it in LCEL through `RunnableLambda`.

## ADR-11: Optional GPU for the local ONNX models; CPU in deployment
On the development laptop's CPU, `bge-reranker-base` took 13–16 s to score 20 query-chunk pairs, which was too slow for an interactive app. On the laptop's GTX 1650 through the ONNX Runtime CUDA provider it takes 0.65 s, and bge-small embedding is 8× faster. GPU and CPU embeddings agree to cosine ≥ 0.999998, so an index built on either device serves queries from the other. The ONNX runtime comes from mutually exclusive uv extras (`cpu`: `fastembed`; `gpu`: `fastembed-gpu` plus CUDA 13 and cuDNN 9 `nvidia-*` wheels) because `onnxruntime` and `onnxruntime-gpu` can't be installed together. `ONNX_DEVICE=auto` picks CUDA when available and logs the provider actually in use, since ONNX Runtime otherwise falls back to CPU silently. The Docker image and VM stay CPU-only, which is why the reranker choice in evaluation weighs CPU latency too.

## ADR-12: JSON mode + Pydantic validation + one repair retry for structured output
The research pipeline and the router need structured output from open models. Tool calling (`with_structured_output`) is the usual choice, but open models served on Groq break tool-call formats more often than plain JSON. JSON mode (`response_format=json_object`) plus Pydantic validation is simpler and works the same way with LangChain's fake chat models in tests. A validation failure is sent back to the model once. If it fails again, the caller degrades gracefully: the router falls back to keywords, and the research answer keeps its evidence and shows the raw text. IDs the model writes in different formats ("e1", "[S1]", "【S1】") are normalised and then checked against what actually exists.

## ADR-13: Router decision rule for "limitations" questions
The ambiguous case is limitations. The rule is that questions about what the **authors state** ("What limitations did the authors identify?", "What future work do they mention?") are evidence questions. Requests for an **assessment** ("What are the limitations of these approaches?", gaps, improvements, adaptations, experiments) are research questions. This follows the examples in the project brief. The rule is written into the router's few-shot prompt and into the keyword fallback, and it's tested on a labelled router set (see docs/evaluation.md).

## ADR-14: A fixed auxiliary retrieval query for the research assistant
A research question like "how could this be improved?" doesn't share vocabulary with the passages where papers discuss their own weaknesses, so retrieving with the user's question alone tends to miss them. The research pipeline therefore runs the shared retriever a second time with a **fixed** query ("limitations, weaknesses, assumptions, failure cases, future work, open problems"), then merges the two ranked lists. It's deterministic and costs one extra retrieval (no LLM call), so it isn't LLM query decomposition.

## ADR-15: Background ingestion with FastAPI BackgroundTasks, plus startup reconciliation
Indexing a paper takes seconds to a minute (parsing, then embedding on CPU or GPU), so `POST /papers/upload` returns `202` straight away and the client polls the paper's status. Ingestion runs in FastAPI's `BackgroundTasks`, in the same process as the API, rather than in a job queue (Celery or RQ with Redis), which would add two services to a single-user student project. The cost is that a task dies with its process. That's handled explicitly: on startup, any paper still marked `processing` becomes `failed` and can be re-uploaded. Because the paper ID is the content hash, re-uploading is idempotent. If the app ever needed many concurrent uploads or several API workers, a real queue would be the next step.

## ADR-16: Repository interface for paper records
Services depend on a `PaperRepository` protocol (`get`, `list_all`, `add`, `update`, `delete`), not on a storage technology. Phase 4 used a JSON file with a lock and atomic replace; Phase 5 swapped in PostgreSQL behind the same interface without changing routes or services. The contract tests run against the JSON registry, SQLite and real PostgreSQL, so all three must behave identically.

## ADR-17: Store the complete structured answer with each message
Conversation history could store only the answer text, but then a UI could not re-render a past research answer with its evidence, citations and directions. Each assistant message therefore keeps the full response payload (JSONB), built by the same function that builds the `/query` response, so the two can't drift apart. The cost is storage (a research answer is about 20 KB), which is irrelevant at this scale. The exchange (question and answer) is written in one transaction and only after the answer succeeds, so history never contains a question without its answer. History is shown to users but never given to the model: conversational memory stays out of scope.

## ADR-18: SQLAlchemy 2.0 + Alembic; migrations at startup; pg8000 driver
Schema changes go through Alembic migrations (generated from the models and reviewed by hand, tested by running them up and down on real PostgreSQL), not `create_all`. The API applies pending migrations when it starts, which is idempotent and what a single-container deployment needs. The driver is pg8000, which is pure Python: the compiled psycopg binary was blocked by the Windows application-control policy on the development machine, and a pure-Python driver also means no native libraries in the Docker image. SQLAlchemy keeps the code driver-independent. JSON columns use JSONB on PostgreSQL and plain JSON elsewhere so fast unit tests can run on SQLite.

## ADR-19: Frontend: plain React + fetch, same-origin `/api` proxy, answers rendered from structured JSON
The UI has a handful of screens' worth of state, so `useState` in one component is enough: no Redux or query library to explain. All network calls live in one file (`api.ts`), which turns FastAPI's two error shapes into readable messages. The browser calls `/api/...` and a proxy forwards it (Vite in development, nginx in production), which removes CORS configuration and means the app never hardcodes a backend address. Research answers are rendered from the structured JSON rather than from text, which is what makes the evidence / analysis / directions separation visible and lets "based on E2" be a working link. Evidence answers are markdown (models produce tables and bold), rendered with `react-markdown`, with citation markers turned into clickable chips. Tests include real backend responses, so drift between the backend schemas and `types.ts` is caught.
