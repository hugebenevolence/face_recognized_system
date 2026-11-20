from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class FaceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    image_path: str
    det_score: float
    model: str = "buffalo_l"
    crop_path: Optional[str] = None
    meta: Optional[dict[str, Any]] = None
    created_at: Optional[datetime] = None


class PersonOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    full_name: str
    person_code: str
    notes: Optional[str] = None
    created_at: datetime
    faces: list[FaceOut] = []


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    ts: datetime
    status: Literal["known", "unknown"]
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    person_code: Optional[str] = None
    confidence: float
    camera_id: str
    track_id: Optional[int] = None
    snapshot_path: Optional[str] = None
    has_evidence: bool = False


class EventDetail(EventOut):
    evidence: Optional[dict[str, Any]] = None


class EventPage(BaseModel):
    items: list[EventOut]
    total: int
    page: int
    page_size: int


class EventIn(BaseModel):
    """Payload the pipeline posts to /internal/events."""

    ts: Optional[datetime] = None
    status: Literal["known", "unknown"]
    person_id: Optional[int] = None
    confidence: float = Field(ge=-1.0, le=1.0)
    camera_id: str
    track_id: Optional[int] = None
    snapshot_path: Optional[str] = None
    evidence: Optional[dict[str, Any]] = None
