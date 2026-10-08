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


# ---- database fixtures -------------------------------------------------------------------------


@pytest.fixture
def sqlite_sessions(tmp_path):
    """SQLite database built by the real Alembic migration (so migrations are tested too)."""
    from researchhelp.db.migrate import upgrade
    from researchhelp.db.session import make_engine, make_session_factory

    url = f"sqlite:///{tmp_path / 'app.db'}"
    upgrade(url)
    engine = make_engine(url)
    yield make_session_factory(engine)
    engine.dispose()


@pytest.fixture
def pg_sessions():
    """Real PostgreSQL from docker compose, in a throwaway database that is migrated fresh for
    each test. Skipped (not failed) when PostgreSQL is not reachable."""
    import sqlalchemy as sa

    from researchhelp.config import get_settings
    from researchhelp.db.migrate import downgrade, upgrade
    from researchhelp.db.session import make_engine, make_session_factory

    base_url = sa.make_url(get_settings().database_url)
    try:
        admin = sa.create_engine(base_url, isolation_level="AUTOCOMMIT")
        with admin.connect() as conn:
            exists = conn.execute(
                sa.text("select 1 from pg_database where datname='researchhelp_test'")
            ).first()
            if not exists:
                conn.execute(sa.text("create database researchhelp_test"))
    except Exception as exc:
        pytest.skip(
            f"PostgreSQL not reachable ({type(exc).__name__}); run: docker compose up -d postgres"
        )
    url = base_url.set(database="researchhelp_test").render_as_string(hide_password=False)
    downgrade_needed = sa.inspect(sa.create_engine(url)).has_table("alembic_version")
    if downgrade_needed:
        downgrade(url, "base")
    upgrade(url)
    engine = make_engine(url)
    yield make_session_factory(engine)
    engine.dispose()
    admin.dispose()
