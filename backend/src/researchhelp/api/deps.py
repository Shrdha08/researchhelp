"""FastAPI dependencies: services live on ``app.state`` (created once in the lifespan, or
injected by tests) and routes receive them through ``Depends``."""

from fastapi import Request

from researchhelp.services.paper_service import PaperService
from researchhelp.services.query_service import QueryService


def paper_service(request: Request) -> PaperService:
    return request.app.state.paper_service


def query_service(request: Request) -> QueryService:
    return request.app.state.query_service
