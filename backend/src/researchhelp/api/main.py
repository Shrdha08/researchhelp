"""FastAPI application.

    uv run uvicorn researchhelp.api.main:app --reload      (or: uv run researchhelp serve)
    Swagger UI: http://localhost:8000/docs

The API layer only does HTTP: validation, status codes, file upload handling and background
scheduling. Application logic lives in ``researchhelp.services``; RAG logic in ``rag``/``graph``.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from researchhelp.api.routes import health, papers, query
from researchhelp.config import get_settings
from researchhelp.services.errors import (
    InvalidUpload,
    LLMUnavailable,
    PaperBusy,
    PaperNotFound,
    PaperNotReady,
)
from researchhelp.services.paper_service import PaperService
from researchhelp.services.query_service import QueryService

log = logging.getLogger(__name__)

_STATUS = {
    InvalidUpload: 400,
    PaperNotFound: 404,
    PaperNotReady: 409,
    PaperBusy: 409,
    LLMUnavailable: 503,
}


def _default_services() -> tuple[PaperService, QueryService, bool]:
    """Production wiring: Qdrant + local models + JSON registry + the LangGraph app."""
    from researchhelp.graph.factory import get_app
    from researchhelp.repository.json_repository import JsonPaperRepository
    from researchhelp.retrieval.factory import get_store

    settings = get_settings()
    papers_svc = PaperService(
        repo=JsonPaperRepository(settings.registry_path),
        store=get_store(),  # loads the embedding models once, at startup
        upload_dir=settings.upload_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_upload_bytes=settings.max_upload_mb * 1024 * 1024,
    )
    stats = papers_svc.reconcile()
    log.info("paper registry reconciled with the index: %s", stats)
    query_svc = QueryService(get_app, papers_svc)
    llm_configured = bool(settings.groq_api_key)
    if llm_configured:
        query_svc.graph()  # warm up: load the reranker and build the graph before first request
    return papers_svc, query_svc, llm_configured


def create_app(
    paper_service: PaperService | None = None,
    query_service: QueryService | None = None,
    llm_configured: bool = True,
) -> FastAPI:
    """Pass services to run with test doubles; omit them for the production wiring."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if paper_service is None or query_service is None:
            (
                app.state.paper_service,
                app.state.query_service,
                app.state.llm_configured,
            ) = _default_services()
        else:
            app.state.paper_service = paper_service
            app.state.query_service = query_service
            app.state.llm_configured = llm_configured
        yield

    app = FastAPI(
        title="ResearchHelp API",
        version="0.4.0",
        description="Upload research papers, ask evidence questions with page citations, "
        "or get research assistance (evidence / analysis / proposed directions).",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for error, code in _STATUS.items():

        async def handler(request: Request, exc: Exception, code: int = code) -> JSONResponse:
            return JSONResponse(status_code=code, content={"detail": str(exc)})

        app.add_exception_handler(error, handler)

    app.include_router(health.router)
    app.include_router(papers.router)
    app.include_router(query.router)
    return app


app = create_app()
