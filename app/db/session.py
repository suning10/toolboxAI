"""
SQLAlchemy engine/session setup. SQLite file by default - fine for the
single-process deployment this app targets; point DATABASE_URL (see
app/config.py) at Postgres if you scale beyond one instance.
"""
import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import DATABASE_URL

# SQLite doesn't create its own parent directory - needed since data/ is
# gitignored and won't exist yet on a fresh clone.
if DATABASE_URL.startswith("sqlite:///"):
    db_dir = os.path.dirname(DATABASE_URL.removeprefix("sqlite:///"))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables that don't exist yet. Called once at app startup
    (see app/main.py). Fine at this scale; move to Alembic migrations
    once the schema needs versioned changes instead of just additions.
    """
    from app.models import chat_session  # noqa: F401 - registers the table on Base.metadata

    Base.metadata.create_all(bind=engine)
