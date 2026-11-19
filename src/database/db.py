"""SQLAlchemy engine, session, and schema initialization."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import URL, Engine
from sqlalchemy.orm import Session

from src.database.schema import SCHEMA_VERSION, Base, SchemaMeta


def utc_now() -> str:
    """Return the current UTC time in ISO 8601 with a 'Z' suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_engine(db_path: Path) -> Engine:
    """Create an engine for a SQLite database with foreign keys enforced."""
    path = Path(db_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(URL.create("sqlite+pysqlite", database=str(path)))

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()

    return engine


@contextmanager
def connect(db_path: Path) -> Iterator[Session]:
    """Yield a transaction-scoped session that commits or rolls back and then closes."""
    engine = make_engine(db_path)
    try:
        with Session(engine, expire_on_commit=False, close_resets_only=False) as session:
            with session.begin():
                yield session
    finally:
        engine.dispose()


def init_db(db_path: Path) -> Path:
    """Create missing tables without changing existing rows or schema names."""
    engine = make_engine(db_path)
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session, session.begin():
            meta = session.get(SchemaMeta, "schema_version")
            if meta is None:
                session.add(SchemaMeta(key="schema_version", value=str(SCHEMA_VERSION)))
            else:
                meta.value = str(SCHEMA_VERSION)
    finally:
        engine.dispose()
    return Path(db_path)


def schema_version(db_path: Path) -> int | None:
    with connect(db_path) as session:
        meta = session.get(SchemaMeta, "schema_version")
        return int(meta.value) if meta else None
