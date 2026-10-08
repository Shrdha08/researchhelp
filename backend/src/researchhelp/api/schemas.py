"""Request/response models. They define the public API contract (and the Swagger docs) and are
deliberately separate from internal dataclasses, so internals can change without breaking clients.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from researchhelp.repository.base import PaperRecord


class _Out(BaseModel):
    model_config = ConfigDict(extra="ignore")


# ----- papers ---------------------------------------------------------------------------------


class PaperOut(_Out):
    id: str
    filename: str
    status: Literal["processing", "ready", "failed"]
    title: str | None = None
    num_pages: int | None = None
    num_chunks: int | None = None
    error: str | None = None
    created_at: str
    updated_at: str

    @classmethod
    def of(cls, record: PaperRecord) -> "PaperOut":
        return cls.model_validate(record.to_dict())


class UploadedPaper(_Out):
    paper: PaperOut
    duplicate: bool = Field(description="True if this exact PDF was already uploaded")


class UploadResponse(_Out):
    papers: list[UploadedPaper]


# ----- query ----------------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    paper_ids: list[str] = Field(min_length=1, max_length=10)
    mode: Literal["auto", "evidence", "research"] = Field(
        "auto", description="auto lets the router decide; evidence/research force a pipeline"
    )
    conversation_id: str | None = Field(
        None,
        description="Add to an existing conversation; omit to start a new one. History is only "
        "stored and displayed: it is not given to the model as context.",
    )


class CitationOut(_Out):
    source_id: str
    paper_id: str
    paper_title: str
    page: int
    section: str
    chunk_id: str
    snippet: str


class SourceOut(_Out):
    paper_id: str
    paper_title: str
    page: int
    section: str
    chunk_id: str
    text: str
    score: float | None = None


class EvidenceResult(_Out):
    answer: str
    citations: list[CitationOut]
    sources: list[SourceOut]


class ResearchEvidenceOut(_Out):
    id: str
    claim: str
    paper_title: str
    citations: list[CitationOut]


class AnalysisOut(_Out):
    statement: str
    based_on: list[str]
    confidence: str


class DirectionOut(_Out):
    title: str
    rationale: str
    based_on: list[str]
    validation_experiment: str


class ResearchResult(_Out):
    status: Literal["ok", "no_evidence", "unstructured"]
    message: str = ""
    evidence: list[ResearchEvidenceOut]
    analysis: list[AnalysisOut]
    directions: list[DirectionOut]
    sources: list[SourceOut]
    raw_text: str = ""


class QueryResponse(_Out):
    conversation_id: str | None = Field(None, description="Conversation this exchange was saved to")
    intent: Literal["evidence", "research"]
    route_source: Literal["llm", "keyword", "forced"]
    route_reason: str
    evidence: EvidenceResult | None = Field(None, description="Set when intent is evidence")
    research: ResearchResult | None = Field(None, description="Set when intent is research")

    @classmethod
    def of(cls, payload: dict, conversation_id: str | None) -> "QueryResponse":
        return cls.model_validate({**payload, "conversation_id": conversation_id})


# ----- conversations ----------------------------------------------------------------------------


class MessageOut(_Out):
    id: int
    role: Literal["user", "assistant"]
    content: str
    created_at: str
    intent: Literal["evidence", "research"] | None = None
    mode: Literal["auto", "evidence", "research"] | None = None
    paper_ids: list[str] = []
    payload: QueryResponse | None = Field(
        None, description="Assistant messages: the full structured answer, as /query returned it"
    )


class ConversationSummary(_Out):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class ConversationOut(ConversationSummary):
    messages: list[MessageOut]


class HealthResponse(_Out):
    status: Literal["ok", "degraded"]
    qdrant: bool
    database: bool
    papers: int
    llm_configured: bool
