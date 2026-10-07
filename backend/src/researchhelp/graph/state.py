"""State passed between LangGraph nodes."""

from typing import Any, Literal, TypedDict

Mode = Literal["auto", "evidence", "research"]
Intent = Literal["evidence", "research"]


class GraphState(TypedDict, total=False):
    # inputs
    question: str
    paper_ids: list[str]
    paper_titles: dict[str, str]
    mode: Mode  # "auto" lets the classifier decide; otherwise the user forces the pipeline
    # routing
    intent: Intent
    route_reason: str
    route_source: Literal["llm", "keyword", "forced"]
    # output: EvidenceAnswer or ResearchAnswer
    result: Any
