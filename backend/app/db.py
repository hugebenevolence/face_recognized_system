from collections.abc import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Iterator[Session]:
    with SessionLocal() as db:
        yield db


def init_db() -> None:
    # Import models so they register on Base.metadata before create_all.
    from . import models  # noqa: F401

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    # create_all never alters existing tables. Until real migrations exist, add the columns
    # introduced after the first release so existing databases keep working (idempotent).
    with engine.begin() as conn:
        for stmt in (
            "ALTER TABLE face_embeddings ADD COLUMN IF NOT EXISTS model VARCHAR(32) NOT NULL DEFAULT 'buffalo_l'",
            "ALTER TABLE face_embeddings ADD COLUMN IF NOT EXISTS meta JSONB",
            "ALTER TABLE face_embeddings ADD COLUMN IF NOT EXISTS crop_path VARCHAR(512)",
            "ALTER TABLE recognition_events ADD COLUMN IF NOT EXISTS evidence JSONB",
            "CREATE INDEX IF NOT EXISTS ix_face_embeddings_model ON face_embeddings (model)",
        ):
            conn.execute(text(stmt))
