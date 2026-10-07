"""Paper registry: application-level records of uploaded papers (status, title, counts).

Qdrant holds the chunks; the registry holds what the application needs to know about each
paper. The interface is what services depend on, so Phase 5 can replace the JSON
implementation with PostgreSQL without touching the API or the services.
"""

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol

PaperStatus = Literal["processing", "ready", "failed"]


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class PaperRecord:
    id: str  # sha256 prefix of the PDF: same value as Qdrant's paper_id
    filename: str
    status: PaperStatus = "processing"
    title: str | None = None
    num_pages: int | None = None
    num_chunks: int | None = None
    error: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict:
        return asdict(self)


class PaperRepository(Protocol):
    def get(self, paper_id: str) -> PaperRecord | None: ...

    def list_all(self) -> list[PaperRecord]: ...

    def add(self, record: PaperRecord) -> PaperRecord: ...

    def update(self, paper_id: str, **changes) -> PaperRecord: ...

    def delete(self, paper_id: str) -> None: ...
