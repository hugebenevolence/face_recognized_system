import json
from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from .. import pipeline_client
from ..config import settings
from ..db import get_db
from ..models import FaceEmbedding, RecognitionEvent
from ..schemas import EventDetail, EventOut, EventPage

router = APIRouter(prefix="/api/events", tags=["events"])


@router.get("", response_model=EventPage)
def list_events(
    q: Optional[str] = Query(None, description="Search person name / ID"),
    person_id: Optional[int] = None,
    status: Optional[Literal["known", "unknown"]] = None,
    date_from: Optional[datetime] = Query(None, alias="from"),
    date_to: Optional[datetime] = Query(None, alias="to"),
    camera_id: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    conds = []
    if q:
        like = f"%{q}%"
        conds.append(
            or_(
                RecognitionEvent.person_name.ilike(like),
                RecognitionEvent.person_code.ilike(like),
            )
        )
    if person_id is not None:
        conds.append(RecognitionEvent.person_id == person_id)
    if status:
        conds.append(RecognitionEvent.status == status)
    if date_from:
        conds.append(RecognitionEvent.ts >= date_from)
    if date_to:
        conds.append(RecognitionEvent.ts <= date_to)
    if camera_id:
        conds.append(RecognitionEvent.camera_id == camera_id)

    total = db.scalar(select(func.count()).select_from(RecognitionEvent).where(*conds)) or 0
    rows = db.scalars(
        select(RecognitionEvent)
        .where(*conds)
        .order_by(RecognitionEvent.ts.desc(), RecognitionEvent.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return EventPage(
        items=[EventOut.model_validate(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


def _event_or_404(db: Session, event_id: int) -> RecognitionEvent:
    ev = db.get(RecognitionEvent, event_id)
    if ev is None:
        raise HTTPException(404, "Event not found")
    return ev


@router.get("/{event_id}", response_model=EventDetail)
def get_event(event_id: int, db: Session = Depends(get_db)):
    """One event with its evidence (candidates, landmarks, quality, model, threshold)."""
    return EventDetail.model_validate(_event_or_404(db, event_id), from_attributes=True)


@router.get("/{event_id}/explain")
def explain_event(event_id: int, refresh: bool = False, db: Session = Depends(get_db)):
    """Occlusion-sensitivity maps for one event, computed from the stored crops (never from
    the live stream) and cached. Explains the match against the recognised person, or for an
    Unknown event against the closest registered person."""
    ev = _event_or_404(db, event_id)
    evidence = ev.evidence
    if not evidence or not evidence.get("aligned_path"):
        raise HTTPException(404, "No evidence stored for this event")

    cache = settings.data_dir / "xai" / "v2" / f"{event_id}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())

    try:
        aligned = (settings.data_dir / evidence["aligned_path"]).read_bytes()
    except OSError:
        raise HTTPException(404, "The stored face crop is missing") from None
    crop = None
    if ev.snapshot_path:
        try:
            crop = (settings.data_dir / ev.snapshot_path).read_bytes()
        except OSError:
            crop = None

    top = evidence.get("top") or []
    target_id = ev.person_id if ev.person_id is not None else (top[0]["person_id"] if top else None)
    refs: list[list[float]] = []
    if target_id is not None:
        rows = db.scalars(
            select(FaceEmbedding.embedding).where(
                FaceEmbedding.person_id == target_id, FaceEmbedding.model == evidence["model"]
            )
        ).all()
        refs = [[float(x) for x in r] for r in rows]
    target_name = ev.person_name or next((t["name"] for t in top if t["person_id"] == target_id), None)
    target_code = ev.person_code or next((t["code"] for t in top if t["person_id"] == target_id), None)

    payload = {
        "model": evidence["model"],
        "refs": refs,
        "bbox": evidence.get("bbox"),
        "kps": evidence.get("kps"),
    }
    try:
        result = pipeline_client.explain(aligned, crop, payload)
    except pipeline_client.PipelineConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except (pipeline_client.NoFaceError, pipeline_client.PipelineUnavailable) as exc:
        raise HTTPException(503, str(exc)) from exc

    out = {
        "event_id": event_id,
        "target": {
            "person_id": target_id,
            "name": target_name,
            "code": target_code,
            "matched": ev.status == "known",
            "available": bool(refs),
        },
        **result,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out
