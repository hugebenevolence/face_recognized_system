from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base

EMBEDDING_DIM = 512  # ArcFace (insightface buffalo_*)


class Person(Base):
    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(primary_key=True)
    full_name: Mapped[str] = mapped_column(String(200))
    person_code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    faces: Mapped[list["FaceEmbedding"]] = relationship(
        back_populates="person", cascade="all, delete-orphan", order_by="FaceEmbedding.id"
    )


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("persons.id", ondelete="CASCADE"), index=True)
    embedding = mapped_column(Vector(EMBEDDING_DIM))
    image_path: Mapped[str] = mapped_column(String(512))
    det_score: Mapped[float] = mapped_column(Float)
    # Embeddings from different model packs are not comparable, so each row records which
    # model produced it; matching only uses rows of the model currently loaded.
    model: Mapped[str] = mapped_column(
        String(32), default="buffalo_l", server_default="buffalo_l", index=True
    )
    # Registration report: chosen face box, landmarks, quality checks, photo size.
    meta = mapped_column(JSONB, nullable=True)
    crop_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)  # aligned 112x112
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    person: Mapped[Person] = relationship(back_populates="faces")

    __table_args__ = (
        Index(
            "ix_face_embeddings_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class RecognitionEvent(Base):
    __tablename__ = "recognition_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)  # known | unknown
    # person_id goes NULL if the person is deleted; name/code are denormalised
    # so history stays readable afterwards.
    person_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("persons.id", ondelete="SET NULL"), nullable=True, index=True
    )
    person_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    person_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    camera_id: Mapped[str] = mapped_column(String(64), index=True)
    track_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    snapshot_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    # Why this was decided: model, threshold, top candidates, landmarks, quality, aligned crop.
    evidence = mapped_column(JSONB, nullable=True)

    @property
    def has_evidence(self) -> bool:
        return self.evidence is not None
