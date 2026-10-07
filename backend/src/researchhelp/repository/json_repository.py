"""JSON-file paper registry (Phase 4 stand-in for PostgreSQL).

Thread-safe within one process: background ingestion threads and request threads update the
registry concurrently, so every read-modify-write holds a lock, and the file is replaced
atomically (write to a temp file, then rename) so a crash never leaves half-written JSON.
Not safe across multiple worker processes; that is one reason Phase 5 moves to PostgreSQL.
"""

import json
import os
import threading
from dataclasses import replace
from pathlib import Path

from researchhelp.repository.base import PaperRecord, utc_now


class JsonPaperRepository:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def _load(self) -> dict[str, PaperRecord]:
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return {pid: PaperRecord(**rec) for pid, rec in raw.items()}

    def _save(self, records: dict[str, PaperRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({pid: r.to_dict() for pid, r in records.items()}, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    def get(self, paper_id: str) -> PaperRecord | None:
        with self._lock:
            return self._load().get(paper_id)

    def list_all(self) -> list[PaperRecord]:
        with self._lock:
            return sorted(self._load().values(), key=lambda r: r.created_at)

    def add(self, record: PaperRecord) -> PaperRecord:
        with self._lock:
            records = self._load()
            if record.id in records:
                raise KeyError(f"paper {record.id} already registered")
            records[record.id] = record
            self._save(records)
            return record

    def update(self, paper_id: str, **changes) -> PaperRecord:
        with self._lock:
            records = self._load()
            if paper_id not in records:
                raise KeyError(paper_id)
            records[paper_id] = replace(records[paper_id], **changes, updated_at=utc_now())
            self._save(records)
            return records[paper_id]

    def delete(self, paper_id: str) -> None:
        with self._lock:
            records = self._load()
            if records.pop(paper_id, None) is not None:
                self._save(records)
