"""Paper lifecycle: upload -> processing -> ready | failed -> delete.

The registry status is the source of truth for the application. Qdrant holds the chunks and is
kept consistent with it: a failed ingestion removes any partial points, a delete removes points
before the record, and ``reconcile()`` repairs both sides at startup.
"""

import hashlib
import logging
from pathlib import Path

from researchhelp.ingestion.metadata.paper_meta import paper_id_from_sha256
from researchhelp.ingestion.pipeline import ingest_pdf
from researchhelp.repository.base import PaperRecord, PaperRepository
from researchhelp.retrieval.vectorstore import PaperVectorStore
from researchhelp.services.errors import (
    InvalidUpload,
    PaperBusy,
    PaperNotFound,
    PaperNotReady,
)

log = logging.getLogger(__name__)
CLI_FILENAME = "(indexed outside the API)"


class PaperService:
    def __init__(
        self,
        repo: PaperRepository,
        store: PaperVectorStore,
        upload_dir: Path,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        max_upload_bytes: int = 50 * 1024 * 1024,
    ):
        self.repo = repo
        self.store = store
        self.upload_dir = Path(upload_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_upload_bytes = max_upload_bytes

    # ----- upload + ingestion ------------------------------------------------------------------

    def validate_upload(self, filename: str, data: bytes) -> None:
        if not filename.lower().endswith(".pdf"):
            raise InvalidUpload(f"{filename}: only .pdf files are accepted")
        if not data:
            raise InvalidUpload(f"{filename}: file is empty")
        if len(data) > self.max_upload_bytes:
            limit = self.max_upload_bytes // (1024 * 1024)
            raise InvalidUpload(f"{filename}: larger than {limit} MB")
        if not data.startswith(b"%PDF"):
            raise InvalidUpload(f"{filename}: not a PDF (missing %PDF header)")

    def register_upload(self, filename: str, data: bytes) -> tuple[PaperRecord, bool]:
        """Store the file and create a ``processing`` record. Returns (record, needs_ingestion).

        The paper ID is the content hash, so uploading the same PDF twice returns the existing
        record instead of indexing a duplicate. A previously failed paper is retried."""
        self.validate_upload(filename, data)
        paper_id = paper_id_from_sha256(hashlib.sha256(data).hexdigest())
        existing = self.repo.get(paper_id)
        if existing and existing.status != "failed":
            return existing, False

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.path_for(paper_id).write_bytes(data)
        if existing:  # retry after failure
            return self.repo.update(paper_id, status="processing", error=None), True
        return self.repo.add(PaperRecord(id=paper_id, filename=filename)), True

    def ingest(self, paper_id: str) -> PaperRecord:
        """Parse, embed and index a registered paper (runs in a background task)."""
        try:
            parsed = ingest_pdf(
                self.path_for(paper_id), self.store, self.chunk_size, self.chunk_overlap
            )
            if parsed.paper_id != paper_id:  # same bytes -> same hash; guards against misuse
                raise RuntimeError("stored file does not match its paper id")
            return self.repo.update(
                paper_id,
                status="ready",
                title=parsed.title,
                num_pages=parsed.num_pages,
                num_chunks=len(parsed.chunks),
                error=None,
            )
        except Exception as exc:
            log.exception("ingestion failed for %s", paper_id)
            try:
                self.store.delete_paper(paper_id)  # no partial index left behind
            except Exception:
                log.exception("could not remove partial points for %s", paper_id)
            return self.repo.update(paper_id, status="failed", error=str(exc)[:500])

    def path_for(self, paper_id: str) -> Path:
        return self.upload_dir / f"{paper_id}.pdf"

    # ----- queries on the registry ---------------------------------------------------------------

    def list_all(self) -> list[PaperRecord]:
        return self.repo.list_all()

    def get(self, paper_id: str) -> PaperRecord:
        record = self.repo.get(paper_id)
        if record is None:
            raise PaperNotFound(f"paper {paper_id} not found")
        return record

    def ready_titles(self, paper_ids: list[str]) -> dict[str, str]:
        """Validate a query scope: every paper must exist and be ready."""
        titles: dict[str, str] = {}
        for pid in dict.fromkeys(paper_ids):
            record = self.get(pid)
            if record.status != "ready":
                raise PaperNotReady(f"paper {pid} is {record.status}, not ready")
            titles[pid] = record.title or record.filename
        return titles

    def delete(self, paper_id: str) -> None:
        record = self.get(paper_id)
        if record.status == "processing":
            raise PaperBusy(f"paper {paper_id} is still being ingested")
        self.store.delete_paper(paper_id)  # index first: never leave orphaned searchable chunks
        self.path_for(paper_id).unlink(missing_ok=True)
        self.repo.delete(paper_id)

    # ----- startup consistency -------------------------------------------------------------------

    def reconcile(self) -> dict[str, int]:
        """Repair registry/index drift after restarts or CLI use."""
        indexed = {p.paper_id: p for p in self.store.list_papers()}
        stats = {"interrupted": 0, "imported": 0, "missing_chunks": 0}
        for record in self.repo.list_all():
            if record.status == "processing":
                # Background tasks do not survive a restart.
                self.repo.update(
                    record.id, status="failed", error="ingestion interrupted by a restart"
                )
                stats["interrupted"] += 1
            elif record.status == "ready" and record.id not in indexed:
                self.repo.update(record.id, status="failed", error="chunks missing from index")
                stats["missing_chunks"] += 1
        known = {r.id for r in self.repo.list_all()}
        for pid, summary in indexed.items():
            if pid not in known:  # e.g. ingested with `researchhelp ingest`
                self.repo.add(
                    PaperRecord(
                        id=pid,
                        filename=CLI_FILENAME,
                        status="ready",
                        title=summary.title,
                        num_pages=summary.num_pages,
                        num_chunks=summary.num_chunks,
                    )
                )
                stats["imported"] += 1
        return stats
