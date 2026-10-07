"""Schemas for the research-assistant pipeline.

Two layers:
- LLM-facing Pydantic models (``EvidenceExtraction``, ``Synthesis``): what the model must return.
  They are lenient about formatting ("e1", "[S1]") because open models vary; IDs are normalised
  and validated against what actually exists afterwards.
- The validated result (``ResearchAnswer``): evidence with real citations; analysis/directions
  that each link to at least one evidence item.
"""

import re
from dataclasses import asdict, dataclass, field
from typing import Literal

from langchain_core.documents import Document
from pydantic import BaseModel, field_validator

from researchhelp.rag.common.context import Citation

_ID = re.compile(r"([SE])\s*(\d+)", re.IGNORECASE)


def normalize_ids(raw: list[str] | str, prefix: str) -> list[str]:
    """Map 'S1', '[S1]', '【S1】', 's 1' or '1' to 'S1' (for prefix 'S').

    Keeps order and drops duplicates and IDs with the other prefix."""
    items = raw if isinstance(raw, list) else re.split(r"[,;\s]+", str(raw))
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        match = _ID.search(text)
        if match and match.group(1).upper() == prefix:
            ident = f"{prefix}{int(match.group(2))}"
        elif text.isdigit():
            ident = f"{prefix}{int(text)}"
        else:
            continue
        if ident not in out:
            out.append(ident)
    return out


# ---- LLM-facing --------------------------------------------------------------------------------


class ExtractedEvidence(BaseModel):
    claim: str
    paper: str = ""
    sources: list[str] = []


class EvidenceExtraction(BaseModel):
    evidence: list[ExtractedEvidence] = []


class AnalysisItem(BaseModel):
    statement: str
    based_on: list[str] = []
    confidence: Literal["low", "medium", "high"] = "medium"

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence(cls, value):
        value = str(value).strip().lower()
        return value if value in ("low", "medium", "high") else "medium"


class DirectionItem(BaseModel):
    title: str
    rationale: str = ""
    based_on: list[str] = []
    validation_experiment: str = ""


class Synthesis(BaseModel):
    analysis: list[AnalysisItem] = []
    directions: list[DirectionItem] = []


# ---- validated result --------------------------------------------------------------------------

Status = Literal["ok", "no_evidence", "unstructured"]


@dataclass
class EvidenceItem:
    id: str  # E1, E2, ...
    claim: str
    paper_title: str
    citations: list[Citation]


@dataclass
class ResearchAnswer:
    question: str
    paper_ids: list[str]
    status: Status
    evidence: list[EvidenceItem] = field(default_factory=list)
    analysis: list[AnalysisItem] = field(default_factory=list)
    directions: list[DirectionItem] = field(default_factory=list)
    sources: list[Document] = field(default_factory=list)
    raw_text: str = ""  # model output kept when structured parsing failed
    message: str = ""
    # Items the model produced but validation removed (invented sources / ungrounded inference).
    dropped: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "paper_ids": self.paper_ids,
            "status": self.status,
            "message": self.message,
            "evidence": [
                {**asdict(e), "citations": [c.to_dict() for c in e.citations]}
                for e in self.evidence
            ],
            "analysis": [a.model_dump() for a in self.analysis],
            "directions": [d.model_dump() for d in self.directions],
            "sources": [{"text": d.page_content, **d.metadata} for d in self.sources],
            "raw_text": self.raw_text,
            "dropped": self.dropped,
        }
