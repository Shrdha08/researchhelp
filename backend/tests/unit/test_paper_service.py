import pytest

from researchhelp.ingestion.pipeline import ingest_pdf
from researchhelp.repository.base import PaperRecord
from researchhelp.repository.json_repository import JsonPaperRepository
from researchhelp.services.errors import InvalidUpload, PaperBusy, PaperNotReady
from researchhelp.services.paper_service import CLI_FILENAME, PaperService


@pytest.fixture
def service(store, tmp_path):
    return PaperService(JsonPaperRepository(tmp_path / "papers.json"), store, tmp_path / "up")


def test_json_repository_round_trip(tmp_path):
    repo = JsonPaperRepository(tmp_path / "r.json")
    repo.add(PaperRecord(id="a", filename="a.pdf"))
    updated = repo.update("a", status="ready", title="T")
    reopened = JsonPaperRepository(tmp_path / "r.json").get("a")
    assert reopened.status == "ready" and reopened.title == "T"
    assert updated.updated_at >= updated.created_at
    with pytest.raises(KeyError):
        repo.add(PaperRecord(id="a", filename="again.pdf"))
    repo.delete("a")
    assert repo.list_all() == []


@pytest.mark.parametrize(
    ("name", "data", "message"),
    [
        ("a.txt", b"%PDF-1.4", "only .pdf"),
        ("a.pdf", b"", "empty"),
        ("a.pdf", b"hello", "not a PDF"),
        ("a.pdf", b"%PDF" + b"x" * 2048, "larger than"),
    ],
)
def test_validate_upload(store, tmp_path, name, data, message):
    service = PaperService(
        JsonPaperRepository(tmp_path / "p.json"), store, tmp_path, max_upload_bytes=1024
    )
    with pytest.raises(InvalidUpload, match=message):
        service.validate_upload(name, data)


def test_failed_paper_is_retried_on_reupload(service, widget_pdf):
    data = widget_pdf.read_bytes()
    record, needs = service.register_upload("w.pdf", data)
    service.repo.update(record.id, status="failed", error="boom")
    again, needs_again = service.register_upload("w.pdf", data)
    assert needs and needs_again and again.status == "processing" and again.error is None


def test_cannot_delete_while_processing_or_query_unready(service, widget_pdf):
    record, _ = service.register_upload("w.pdf", widget_pdf.read_bytes())
    with pytest.raises(PaperBusy):
        service.delete(record.id)
    with pytest.raises(PaperNotReady):
        service.ready_titles([record.id])


def test_reconcile_repairs_registry_and_index_drift(service, widget_pdf, gadget_pdf):
    gadget = ingest_pdf(gadget_pdf, service.store, 300, 50)  # indexed via CLI, not registered
    service.repo.add(PaperRecord(id="stuck", filename="s.pdf"))  # processing at "crash" time
    service.repo.add(PaperRecord(id="gone", filename="g.pdf", status="ready"))  # no chunks

    stats = service.reconcile()

    assert stats == {"interrupted": 1, "imported": 1, "missing_chunks": 1}
    assert service.get("stuck").status == "failed"
    assert service.get("gone").status == "failed"
    imported = service.get(gadget.paper_id)
    assert (imported.status, imported.filename) == ("ready", CLI_FILENAME)
    assert imported.title == gadget.title
    assert service.reconcile() == {"interrupted": 0, "imported": 0, "missing_chunks": 0}
