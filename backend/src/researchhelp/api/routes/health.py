from fastapi import APIRouter, Request

from researchhelp.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    """Liveness plus dependency checks: Qdrant reachable, number of registered papers, and
    whether an LLM key is configured (queries need it, uploads do not)."""
    papers = request.app.state.paper_service
    try:
        papers.store.client.get_collections()
        qdrant = True
    except Exception:
        qdrant = False
    llm = bool(request.app.state.llm_configured)
    return HealthResponse(
        status="ok" if qdrant and llm else "degraded",
        qdrant=qdrant,
        papers=len(papers.list_all()),
        llm_configured=llm,
    )
