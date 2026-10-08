# Architecture

```
React (Vite)  ──HTTP──►  FastAPI  (thin: validation, file storage, persistence, background ingestion)
                            │
              ┌─────────────┼───────────────────────────┐
              ▼             ▼                           ▼
        PostgreSQL     Ingestion pipeline          LangGraph app
     (papers, status,  PyMuPDF→clean→chunk→        classify ─► evidence_rag ─┐
      conversations,   metadata→embed→Qdrant       (auto/override)            ├─► END
      messages)                                          └────► research_asst ┘
                                                             │           │
                                                             ▼           ▼
                                                   Shared ScopedRetriever (one module)
                                                   paper_ids filter → per-paper fan-out
                                                   → hybrid (dense + BM25 sparse, RRF in Qdrant)
                                                   → merge → cross-encoder rerank → top-N
                                                             │
                                                           Qdrant
                                                             │
                                                     Groq LLM (LangChain)
```

## Layers

| Layer | Responsibility | Package |
|---|---|---|
| Ingestion | PDF → cleaned per-page text → section-tagged chunks → vectors in Qdrant | `researchhelp.ingestion` |
| Retrieval | Scoped (paper_id-filtered) search; fan-out per paper; hybrid + rerank (Phase 2) | `researchhelp.retrieval` |
| RAG pipelines | Evidence Q&A chain and research-assistant chains (LangChain) | `researchhelp.rag` |
| Orchestration | Intent routing (LangGraph, Phase 3) | `researchhelp.graph` |
| API | HTTP interface, no RAG logic (Phase 4) | `researchhelp.api` |
| Persistence | Paper registry, conversations (PostgreSQL, Phase 5) | `researchhelp.db` |

## Ingestion data flow

```
PDF ─► PyMuPDF get_text("dict")              per-page lines + font size/bold (content-stream order)
    ─► cleaning                               ligatures, de-hyphenation, repeated header/footer removal
    ─► section detection                      numbered / known-name headings with larger-or-bold font
    ─► per-page recursive chunking            a chunk never spans two pages → exact page citations
    ─► metadata                               paper_id, paper_title, page, section, chunk_id
    ─► FastEmbed bge-small (dense)            + BM25 sparse vector (used by hybrid search in Phase 2)
    ─► Qdrant upsert                          payload index on paper_id, section
```

## Qdrant point schema

| Field | Type | Notes |
|---|---|---|
| id | UUIDv5(chunk_id) | deterministic → re-ingesting a paper overwrites instead of duplicating |
| vector `dense` | float[384], cosine | bge-small-en-v1.5 |
| vector `sparse` | sparse, IDF modifier | Qdrant/bm25 |
| payload | `paper_id, paper_title, page, section, chunk_id, chunk_index, text` | `paper_id`, `section` are keyword-indexed |

## Query data flow (Phase 1 baseline)

```
question + paper_ids
  ─► ScopedRetriever (semantic): for each paper, top-k with filter paper_id == X, excluding References
  ─► merge by score → top k_final
  ─► format context as [S1] (Paper: title, p.N, §Section) text ...
  ─► Groq LLM with grounding prompt
  ─► parse [S#] markers → resolve to paper/page/section; drop IDs not in context
  ─► answer + citations
```

## Routing (LangGraph, Phase 3)

```
START ──(mode = evidence | research)─────────────────► evidence_rag | research_assistant ─► END
  └──(mode = auto)──► classify ──(intent)────────────► evidence_rag | research_assistant ─► END
                       gpt-oss-20b few-shot JSON;
                       keyword fallback on failure
```

The graph decides only the *intent*. Scope (`paper_ids`) is passed through unchanged, and both pipelines use the same `ScopedRetriever`. The graph state records how each route was chosen: `llm`, `keyword` or `forced`.

## Research-assistant data flow (Phase 3)

```
question + paper_ids
  ─► ScopedRetriever(question)  ┐  merged by rank, de-duplicated,
  ─► ScopedRetriever(fixed      ┘  capped at research_k_final (10)
       "limitations / future work" query)
  ─► Chain A: extract evidence (JSON)       sees the [S#] excerpts; may only restate them
       └─ validate: drop claims without a real [S#] source; number E1..En; attach pages
  ─► Chain B: synthesise (JSON)             sees ONLY the evidence list, never the excerpts
       └─ validate: drop analysis/direction items not linked to a real E#
  ─► ResearchAnswer  Evidence (cited) | Analysis (inferred, with confidence)
                     | Proposed directions (hypotheses + how to test them)
```

With no valid evidence, synthesis is skipped (`status = no_evidence`). If the model's JSON can't be repaired, the evidence is kept and the raw text is shown (`status = unstructured`).

## HTTP API (Phase 4)

```
routes (api/routes/*)          HTTP only: validation (Pydantic), status codes, uploads, background tasks
   │
services (services/*)          paper lifecycle, dedupe, scope validation, consistency; raise ServiceErrors
   │                           (mapped once to 400/404/409/503 in api/main.py)
   ├── PaperRepository         PostgreSQL (repository/sql_repository.py); JSON registry kept for tests
   ├── PaperVectorStore        Qdrant chunks
   └── LangGraph app           evidence / research pipelines (rag/, graph/)
```

| Endpoint | Purpose |
|---|---|
| `POST /papers/upload` | multi-file upload; 202 + `processing`; indexing runs as a background task; same PDF → `duplicate: true` |
| `GET /papers`, `GET /papers/{id}` | registry records: status `processing / ready / failed`, title, pages, chunks, error |
| `DELETE /papers/{id}` | removes Qdrant points, then the file, then the record (409 while processing) |
| `POST /query` | `{question, paper_ids, mode}` → `{intent, route_source, route_reason, evidence | research}` |
| `GET /health` | Qdrant reachable, paper count, LLM key configured |

**Paper status lifecycle:** `processing` → `ready` | `failed`. Failed papers have their partial chunks removed and can be re-uploaded. At startup, papers left in `processing` (their background task died with the process) become `failed`, papers indexed through the CLI are imported, and `ready` records whose chunks are missing become `failed`.

## Data stores: why PostgreSQL and Qdrant (Phase 5)

| | PostgreSQL | Qdrant |
|---|---|---|
| **Holds** | Application records: papers (status, title, counts, errors), conversations, messages (including the full structured answers) | Chunks: text, page, section, dense and BM25 sparse vectors |
| **Good at** | Relational integrity (foreign keys, cascade delete), transactions, queries over records | Filtered nearest-neighbour search, hybrid dense + sparse fusion |
| **Queried by** | Services (paper status, history) | The retriever (every question) |
| **Link** | `papers.id` | `paper_id` in every point's payload (the same content hash) |

Neither is a substitute for the other: PostgreSQL could store vectors (pgvector) but has no built-in sparse+dense fusion, and Qdrant could store a payload per point but has no foreign keys, transactions or cascading deletes. The only place where the two must agree is the paper's lifecycle, and that is the service's job: ingestion writes chunks first and the `ready` status last; deletion removes chunks first and the record last; startup reconciliation repairs any drift.

```
papers (id PK = content hash, filename, status, title, num_pages, num_chunks, error, timestamps)
conversations (id PK uuid, title, created_at, updated_at)
messages (id PK, conversation_id FK → conversations ON DELETE CASCADE, role, content,
          intent, mode, paper_ids JSONB, payload JSONB, created_at)
```

An assistant message's `payload` is the same JSON `/query` returned (route, evidence or research body, citations). Deleting a paper does not touch history: stored answers cite the pages that were true when they were given.

## Frontend (Phase 6)

```
browser ── /api/* ──► Vite dev proxy (dev) | nginx (Phase 7) ──► FastAPI   (prefix stripped)
   │
   App.tsx  state: papers, selection, conversations, thread     api.ts  the only fetch calls
   ├── PaperPanel · QueryBox · ConversationList
   └── EvidenceAnswer · ResearchAnswer · SourcesPanel           types.ts mirrors the backend schemas
```

The UI keeps the separation the backend enforces: a research answer renders as three visually distinct sections (evidence with page citations; analysis labelled as inferred; directions labelled as hypotheses to test), and every analysis or direction item links back to the evidence items it builds on.
