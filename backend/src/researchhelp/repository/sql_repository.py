"""PostgreSQL-backed paper registry (same interface as the Phase 4 JSON registry).

Every operation is its own short transaction, so concurrent request threads and background
ingestion threads need no shared lock: PostgreSQL provides the isolation.
"""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from researchhelp.db.models import Paper
from researchhelp.repository.base import PaperRecord


def _iso(value: datetime) -> str:
    if value.tzinfo is None:  # SQLite returns naive datetimes
        value = value.replace(tzinfo=UTC)
    return value.isoformat(timespec="seconds")


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _record(row: Paper) -> PaperRecord:
    return PaperRecord(
        id=row.id,
        filename=row.filename,
        status=row.status,  # type: ignore[arg-type]
        title=row.title,
        num_pages=row.num_pages,
        num_chunks=row.num_chunks,
        error=row.error,
        created_at=_iso(row.created_at),
        updated_at=_iso(row.updated_at),
    )


class SqlPaperRepository:
    def __init__(self, sessions: sessionmaker[Session]):
        self._sessions = sessions

    def get(self, paper_id: str) -> PaperRecord | None:
        with self._sessions() as s:
            row = s.get(Paper, paper_id)
            return _record(row) if row else None

    def list_all(self) -> list[PaperRecord]:
        with self._sessions() as s:
            rows = s.scalars(select(Paper).order_by(Paper.created_at, Paper.id))
            return [_record(r) for r in rows]

    def add(self, record: PaperRecord) -> PaperRecord:
        with self._sessions.begin() as s:
            s.add(
                Paper(
                    id=record.id,
                    filename=record.filename,
                    status=record.status,
                    title=record.title,
                    num_pages=record.num_pages,
                    num_chunks=record.num_chunks,
                    error=record.error,
                    created_at=_parse(record.created_at),
                    updated_at=_parse(record.updated_at),
                )
            )
            try:
                s.flush()
            except IntegrityError as exc:
                raise KeyError(f"paper {record.id} already registered") from exc
        return record

    def update(self, paper_id: str, **changes) -> PaperRecord:
        with self._sessions.begin() as s:
            row = s.get(Paper, paper_id, with_for_update=True)
            if row is None:
                raise KeyError(paper_id)
            for key, value in changes.items():
                setattr(row, key, value)
            row.updated_at = datetime.now(UTC)
            s.flush()
            return _record(row)

    def delete(self, paper_id: str) -> None:
        with self._sessions.begin() as s:
            row = s.get(Paper, paper_id)
            if row is not None:
                s.delete(row)
