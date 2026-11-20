import base64
import json
import shutil
import uuid
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import or_, select
from sqlalchemy.orm import Session, aliased, selectinload

from .. import pipeline_client
from ..config import settings
from ..db import get_db
from ..models import FaceEmbedding, Person
from ..schemas import PersonOut

router = APIRouter(prefix="/api/persons", tags=["persons"])

ALLOWED_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}

Item = tuple[bytes, str, dict]  # (original bytes, file extension, pipeline /embed result)


# ------------------------------------------------------------------ input handling
def _parse_boxes(raw: Optional[str], n: int) -> list[Optional[list[float]]]:
    """`boxes` = JSON list with one [x1,y1,x2,y2] (image pixels) or null per uploaded image."""
    if not raw:
        return [None] * n
    try:
        boxes = json.loads(raw)
    except ValueError:
        raise HTTPException(422, "boxes must be valid JSON") from None
    if not isinstance(boxes, list) or len(boxes) != n:
        raise HTTPException(422, "boxes needs exactly one entry (or null) per image")
    out: list[Optional[list[float]]] = []
    for b in boxes:
        if b is None:
            out.append(None)
            continue
        try:
            v = [float(x) for x in b]
            if len(v) != 4 or v[2] <= v[0] or v[3] <= v[1]:
                raise ValueError
        except (TypeError, ValueError):
            raise HTTPException(422, "each box must be [x1, y1, x2, y2] with x2>x1 and y2>y1") from None
        out.append(v)
    return out


def _read_and_embed(files: list[UploadFile], boxes: list[Optional[list[float]]]) -> list[Item]:
    """Validate every upload and extract its embedding *before* touching the DB/disk,
    so a bad image leaves no partial registration behind."""
    if not files:
        raise HTTPException(422, "Add at least one photo")
    limit = settings.max_upload_mb * 1024 * 1024
    out: list[Item] = []
    for f, box in zip(files, boxes):
        ext = ALLOWED_TYPES.get(f.content_type or "")
        if ext is None:
            raise HTTPException(422, "Unsupported file type. Use a JPEG or PNG or WebP image")
        data = f.file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB")
        try:
            result = pipeline_client.embed_image(data, f.filename or "face", f.content_type or "", box)
        except pipeline_client.NoFaceError as exc:
            raise HTTPException(422, str(exc)) from exc
        except pipeline_client.PipelineUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        out.append((data, ext, result))
    return out


# ------------------------------------------------------------------ trust checks
def _nearest_other(
    db: Session, emb: Any, model: str, exclude_person: Optional[int] = None
) -> Optional[tuple[int, float]]:
    """(person_id, cosine similarity) of the closest stored face of another person."""
    dist = FaceEmbedding.embedding.cosine_distance(emb)
    stmt = select(FaceEmbedding.person_id, (1 - dist).label("sim")).where(FaceEmbedding.model == model)
    if exclude_person is not None:
        stmt = stmt.where(FaceEmbedding.person_id != exclude_person)
    row = db.execute(stmt.order_by(dist).limit(1)).first()
    return (row.person_id, float(row.sim)) if row else None


def _conflict(code: str, message: str, **extra: Any) -> HTTPException:
    return HTTPException(409, detail={"code": code, "message": message, **extra})


def _check_new_faces(db: Session, items: list[Item], *, own_person: Optional[int], force: bool) -> None:
    """Refuse (409, overridable with force=true) enrolments that look like a mistake:
    the same face as someone else, photos of different people mixed together, or a photo
    that does not look like the person it is being added to."""
    if force:
        return
    # 1. photos uploaded together must agree with each other
    for (_, _, a), (_, _, b) in combinations(items, 2):
        sim = sum(x * y for x, y in zip(a["embedding"], b["embedding"]))  # both L2-normalised
        if sim < settings.consistency_threshold:
            raise _conflict(
                "images_disagree",
                f"These photos do not look like the same person. Similarity {sim:.2f}",
                score=sim,
            )
    for _, _, res in items:
        # 2. same face as a different registered person?
        hit = _nearest_other(db, res["embedding"], res["model"], exclude_person=own_person)
        if hit and hit[1] >= settings.dup_threshold:
            p = db.get(Person, hit[0])
            raise _conflict(
                "possible_duplicate",
                f"This face is very similar to {p.full_name} with ID {p.person_code} at {hit[1]:.2f}. "
                "It may be the same person",
                person={"id": p.id, "full_name": p.full_name, "person_code": p.person_code},
                score=hit[1],
            )
        # 3. adding to an existing person: must resemble their current photos
        if own_person is not None:
            dist = FaceEmbedding.embedding.cosine_distance(res["embedding"])
            best = db.scalar(
                select(1 - dist)
                .where(FaceEmbedding.person_id == own_person, FaceEmbedding.model == res["model"])
                .order_by(dist)
                .limit(1)
            )
            if best is not None and float(best) < settings.consistency_threshold:
                raise _conflict(
                    "inconsistent_with_person",
                    f"This photo does not look like the other photos of this person. Similarity {float(best):.2f}",
                    score=float(best),
                )


# ------------------------------------------------------------------ storing
def _store(db: Session, person: Person, items: list[Item]) -> list[Path]:
    """Write images to disk and add embedding rows. Returns written paths for cleanup."""
    written: list[Path] = []
    rel_dir = Path("faces") / str(person.id)
    (settings.data_dir / rel_dir).mkdir(parents=True, exist_ok=True)
    for data, ext, res in items:
        stem = uuid.uuid4().hex
        rel = rel_dir / f"{stem}{ext}"
        (settings.data_dir / rel).write_bytes(data)
        written.append(settings.data_dir / rel)

        crop_rel: Optional[str] = None
        if res.get("aligned_png_b64"):
            crop_rel = (rel_dir / f"{stem}_aligned.png").as_posix()
            (settings.data_dir / crop_rel).write_bytes(base64.b64decode(res["aligned_png_b64"]))
            written.append(settings.data_dir / crop_rel)

        db.add(
            FaceEmbedding(
                person_id=person.id,
                embedding=res["embedding"],
                image_path=rel.as_posix(),
                det_score=res["det_score"],
                model=res["model"],
                crop_path=crop_rel,
                meta={
                    "bbox": res.get("bbox"),
                    "kps": res.get("kps"),
                    "image_size": res.get("image_size"),
                    "faces_in_photo": res.get("face_count"),
                    "quality": res.get("quality"),
                },
            )
        )
    return written


def _commit_with_cleanup(db: Session, written: list[Path]) -> None:
    try:
        db.commit()
    except Exception:
        db.rollback()
        for p in written:
            p.unlink(missing_ok=True)
        raise


# ------------------------------------------------------------------ endpoints
@router.get("", response_model=list[PersonOut])
def list_persons(q: Optional[str] = None, db: Session = Depends(get_db)):
    stmt = select(Person).options(selectinload(Person.faces)).order_by(Person.id.desc())
    if q:
        like = f"%{q}%"
        stmt = stmt.where(or_(Person.full_name.ilike(like), Person.person_code.ilike(like)))
    return db.scalars(stmt).all()


@router.get("/{person_id}", response_model=PersonOut)
def get_person(person_id: int, db: Session = Depends(get_db)):
    person = db.scalar(select(Person).options(selectinload(Person.faces)).where(Person.id == person_id))
    if person is None:
        raise HTTPException(404, "Person not found")
    return person


@router.post("", response_model=PersonOut, status_code=201)
def create_person(
    full_name: str = Form(..., min_length=1, max_length=200),
    person_code: str = Form(..., min_length=1, max_length=64),
    notes: Optional[str] = Form(None),
    images: list[UploadFile] = File(...),
    boxes: Optional[str] = Form(None),
    force: bool = Form(False),
    db: Session = Depends(get_db),
):
    full_name, person_code = full_name.strip(), person_code.strip()
    if not full_name or not person_code:
        raise HTTPException(422, "Name and ID are required")
    if db.scalar(select(Person.id).where(Person.person_code == person_code)):
        raise HTTPException(409, f"Person ID {person_code} already exists")

    items = _read_and_embed(images, _parse_boxes(boxes, len(images)))
    _check_new_faces(db, items, own_person=None, force=force)

    person = Person(full_name=full_name, person_code=person_code, notes=notes or None)
    db.add(person)
    db.flush()  # assigns person.id
    written = _store(db, person, items)
    _commit_with_cleanup(db, written)
    db.refresh(person, ["faces"])
    return person


@router.post("/{person_id}/faces", response_model=PersonOut)
def add_faces(
    person_id: int,
    images: list[UploadFile] = File(...),
    boxes: Optional[str] = Form(None),
    force: bool = Form(False),
    db: Session = Depends(get_db),
):
    person = db.get(Person, person_id)
    if person is None:
        raise HTTPException(404, "Person not found")
    items = _read_and_embed(images, _parse_boxes(boxes, len(images)))
    _check_new_faces(db, items, own_person=person_id, force=force)
    written = _store(db, person, items)
    _commit_with_cleanup(db, written)
    db.refresh(person, ["faces"])
    return person


@router.get("/{person_id}/report")
def registration_report(person_id: int, db: Session = Depends(get_db)):
    """How this person was registered and how well the system can tell them apart from
    everyone else: the data behind the registration-report dialog."""
    person = db.scalar(select(Person).options(selectinload(Person.faces)).where(Person.id == person_id))
    if person is None:
        raise HTTPException(404, "Person not found")

    a, b = aliased(FaceEmbedding), aliased(FaceEmbedding)
    rows = db.execute(
        select(a.id, b.id, (1 - a.embedding.cosine_distance(b.embedding)).label("sim")).where(
            a.person_id == person_id, b.person_id == person_id, a.id < b.id, a.model == b.model
        )
    ).all()
    pairs = [{"a": r[0], "b": r[1], "similarity": float(r[2])} for r in rows]

    nearest = None
    for f in person.faces:
        hit = _nearest_other(db, f.embedding, f.model, exclude_person=person_id)
        if hit and (nearest is None or hit[1] > nearest["score"]):
            other = db.get(Person, hit[0])
            nearest = {
                "person_id": other.id,
                "full_name": other.full_name,
                "person_code": other.person_code,
                "score": hit[1],
            }

    sims = [p["similarity"] for p in pairs]
    return {
        "person": PersonOut.model_validate(person),
        "pairs": pairs,
        "consistency": {"min": min(sims), "mean": sum(sims) / len(sims)} if sims else None,
        "nearest_other": nearest,
        "thresholds": {
            "duplicate": settings.dup_threshold,
            "consistency": settings.consistency_threshold,
        },
    }


@router.delete("/{person_id}", status_code=204)
def delete_person(person_id: int, db: Session = Depends(get_db)):
    person = db.get(Person, person_id)
    if person is None:
        raise HTTPException(404, "Person not found")
    db.delete(person)  # cascades to face_embeddings; events.person_id -> NULL
    db.commit()
    shutil.rmtree(settings.data_dir / "faces" / str(person_id), ignore_errors=True)
