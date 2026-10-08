from fastapi import APIRouter, Request

from researchhelp.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    """Liveness plus dependency checks: Qdrant and PostgreSQL reachable, number of registered
    papers, and whether an LLM key is configured (queries need it, uploads do not)."""
    state = request.app.state
    papers = state.paper_service
    try:
        papers.store.client.get_collections()
        qdrant = True
    except Exception:
        qdrant = False
    try:
        count = len(papers.list_all())
        database = True
    except Exception:  # database down: report it instead of failing the health check itself
        count, database = 0, False
    llm = bool(state.llm_configured)
    return HealthResponse(
        status="ok" if qdrant and database and llm else "degraded",
        qdrant=qdrant,
        database=database,
        papers=count,
        llm_configured=llm,
    )
