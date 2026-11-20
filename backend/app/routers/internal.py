"""Endpoints used only by the pipeline service (shared-secret protected)."""

import secrets

from fastapi import APIRouter, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder

from ..config import settings
from ..db import SessionLocal
from ..models import Person, RecognitionEvent
from ..schemas import EventIn, EventOut
from ..ws import manager

router = APIRouter(prefix="/internal", tags=["internal"], include_in_schema=False)


def _save(payload: EventIn) -> EventOut:
    with SessionLocal() as db:
        ev = RecognitionEvent(
            status=payload.status,
            person_id=payload.person_id,
            confidence=payload.confidence,
            camera_id=payload.camera_id,
            track_id=payload.track_id,
            snapshot_path=payload.snapshot_path,
            evidence=payload.evidence,
        )
        if payload.ts:
            ev.ts = payload.ts
        if payload.person_id is not None:
            person = db.get(Person, payload.person_id)
            if person is None:  # deleted between match and report -> treat as unknown
                ev.person_id, ev.status = None, "unknown"
            else:
                ev.person_name, ev.person_code = person.full_name, person.person_code
        db.add(ev)
        db.commit()
        db.refresh(ev)
        return EventOut.model_validate(ev)


@router.post("/events", status_code=201)
async def ingest_event(payload: EventIn, x_internal_token: str = Header("")):
    if not secrets.compare_digest(x_internal_token, settings.internal_token):
        raise HTTPException(401, "bad token")
    event = await run_in_threadpool(_save, payload)
    await manager.broadcast(jsonable_encoder(event))
    return {"id": event.id}
