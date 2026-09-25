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
PDF ─► PyMuPDF get_text("dict", sort=True)   per-page lines + font size/bold
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
