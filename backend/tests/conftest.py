import pytest
from qdrant_client import QdrantClient

from researchhelp.ingestion.pipeline import ingest_pdf
from researchhelp.retrieval.vectorstore import PaperVectorStore
from tests.fakes import HashingDense, HashingSparse
from tests.pdf_factory import gadget_paper, widget_paper


@pytest.fixture(scope="session")
def pdf_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("pdfs")


@pytest.fixture(scope="session")
def widget_pdf(pdf_dir):
    return widget_paper(pdf_dir / "widget.pdf")


@pytest.fixture(scope="session")
def gadget_pdf(pdf_dir):
    return gadget_paper(pdf_dir / "gadget.pdf")


@pytest.fixture
def store():
    """Empty in-memory Qdrant with the production schema and hashing encoders."""
    s = PaperVectorStore(QdrantClient(":memory:"), HashingDense(), HashingSparse(), "test_chunks")
    s.ensure_collection()
    return s


@pytest.fixture
def indexed(store, widget_pdf, gadget_pdf):
    """Store with both synthetic papers ingested (small chunks -> several per page)."""
    widget = ingest_pdf(widget_pdf, store, chunk_size=300, chunk_overlap=50)
    gadget = ingest_pdf(gadget_pdf, store, chunk_size=300, chunk_overlap=50)
    return store, widget, gadget
