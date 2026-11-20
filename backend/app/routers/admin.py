"""Maintenance: keep stored faces usable after a model change and fill in missing reports."""

import base64
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import pipeline_client
from ..config import settings
from ..db import get_db
from ..models import FaceEmbedding

router = APIRouter(prefix="/api/admin", tags=["admin"])

CONTENT_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def _current_model() -> str:
    try:
        model = pipeline_client.status().get("model")
    except pipeline_client.PipelineUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    if not model:
        raise HTTPException(503, "Recognition service did not report its model")
    return model


def _by_image(db: Session) -> dict[str, list[FaceEmbedding]]:
    groups: dict[str, list[FaceEmbedding]] = {}
    for row in db.scalars(select(FaceEmbedding).order_by(FaceEmbedding.id)):
        groups.setdefault(row.image_path, []).append(row)
    return groups


def _todo(groups: dict[str, list[FaceEmbedding]], current: str) -> list[str]:
    """Images with no embedding for the current model, or whose report data is missing."""
    out = []
    for image_path, rows in groups.items():
        cur = next((r for r in rows if r.model == current), None)
        if cur is None or not (cur.meta and cur.meta.get("bbox")):
            out.append(image_path)
    return out


@router.get("/model-status")
def model_status(db: Session = Depends(get_db)):
    try:
        current = pipeline_client.status().get("model")
    except pipeline_client.PipelineUnavailable:
        current = None
    groups = _by_image(db)
    models: dict[str, int] = {}
    for rows in groups.values():
        for r in rows:
            models[r.model] = models.get(r.model, 0) + 1
    return {
        "current_model": current,
        "embeddings_by_model": models,
        "images_total": len(groups),
        "needs_update": len(_todo(groups, current)) if current else 0,
    }


@router.post("/reanalyze")
def reanalyze(db: Session = Depends(get_db)):
    """Re-run the current model over every stored registration photo: adds embeddings for
    the current model (after switching model pack) and backfills the registration report
    for faces registered before reports existed. Originals are kept, so nothing is lost."""
    current = _current_model()
    groups = _by_image(db)
    done, failed = 0, []
    for image_path in _todo(groups, current):
        rows = groups[image_path]
        cur = next((r for r in rows if r.model == current), None)
        src = settings.data_dir / image_path
        try:
            data = src.read_bytes()
        except OSError:
            failed.append({"image_path": image_path, "error": "file missing"})
            continue
        known_meta = next((r.meta for r in rows if r.meta and r.meta.get("bbox")), None)
        box = known_meta["bbox"] if known_meta else None  # keep selecting the same face
        ctype = CONTENT_TYPES.get(src.suffix.lower(), "image/jpeg")
        try:
            try:
                res = pipeline_client.embed_image(data, src.name, ctype, box)
            except pipeline_client.NoFaceError:
                if box is None:
                    raise
                res = pipeline_client.embed_image(data, src.name, ctype, None)
        except (pipeline_client.NoFaceError, pipeline_client.PipelineUnavailable) as exc:
            failed.append({"image_path": image_path, "error": str(exc)})
            continue

        suffix = "_aligned.png" if cur is not None else f"_aligned_{current}.png"
        crop_rel = (Path(image_path).parent / f"{Path(image_path).stem}{suffix}").as_posix()
        (settings.data_dir / crop_rel).write_bytes(base64.b64decode(res["aligned_png_b64"]))
        meta = {
            "bbox": res.get("bbox"),
            "kps": res.get("kps"),
            "image_size": res.get("image_size"),
            "faces_in_photo": res.get("face_count"),
            "quality": res.get("quality"),
        }
        if cur is None:
            db.add(
                FaceEmbedding(
                    person_id=rows[0].person_id,
                    embedding=res["embedding"],
                    image_path=image_path,
                    det_score=res["det_score"],
                    model=current,
                    meta=meta,
                    crop_path=crop_rel,
                )
            )
        else:
            cur.meta, cur.crop_path, cur.det_score = meta, crop_rel, res["det_score"]
        done += 1
    db.commit()
    return {"model": current, "updated": done, "failed": failed}
