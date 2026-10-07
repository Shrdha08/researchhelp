from typing import Annotated

from fastapi import APIRouter, Depends

from researchhelp.api.deps import query_service
from researchhelp.api.schemas import QueryRequest, QueryResponse
from researchhelp.services.query_service import QueryService

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, service: Annotated[QueryService, Depends(query_service)]):
    """Ask a question about the selected papers.

    The router sends it to evidence Q&A (cited answer) or to the research assistant (evidence /
    analysis / proposed directions); ``mode`` overrides the router. A plain ``def`` endpoint on
    purpose: retrieval and the LLM calls are blocking, so FastAPI runs it in its thread pool."""
    return QueryResponse.of(service.ask(body.question, body.paper_ids, body.mode))
