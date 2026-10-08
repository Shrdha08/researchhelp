"""Engine and session handling."""

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker


def make_engine(url: str) -> Engine:
    """Engine with connection health checks (so a restarted database is picked up again)."""
    engine = create_engine(url, pool_pre_ping=True, future=True)
    if engine.dialect.name == "sqlite":  # tests: enforce foreign keys like PostgreSQL does

        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn, _record):
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)
