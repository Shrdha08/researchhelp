"""Query use case: validate the scope, run the LangGraph app, return its final state.

The graph is created lazily on first use, so the API can start (and /health can report) even
when the LLM is not configured; queries then fail with a clear 503 instead of a crash."""

import threading
from collections.abc import Callable

from researchhelp.rag.common.llm import MissingAPIKeyError
from researchhelp.services.errors import LLMUnavailable
from researchhelp.services.paper_service import PaperService


class QueryService:
    def __init__(self, graph_factory: Callable, papers: PaperService):
        self._graph_factory = graph_factory
        self._graph = None
        self._lock = threading.Lock()
        self.papers = papers

    def graph(self):
        with self._lock:
            if self._graph is None:
                try:
                    self._graph = self._graph_factory()
                except MissingAPIKeyError as exc:
                    raise LLMUnavailable(str(exc)) from exc
            return self._graph

    def ask(self, question: str, paper_ids: list[str], mode: str = "auto") -> dict:
        titles = self.papers.ready_titles(paper_ids)
        return self.graph().invoke(
            {"question": question, "paper_ids": list(titles), "paper_titles": titles, "mode": mode}
        )
