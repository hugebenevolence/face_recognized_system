"""Image analysis for the registration UI (group-photo picker, quality report)."""

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .. import pipeline_client
from ..config import settings

router = APIRouter(prefix="/api/faces", tags=["faces"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _read_image(file: UploadFile) -> bytes:
    if (file.content_type or "") not in ALLOWED_TYPES:
        raise HTTPException(422, "Unsupported file type. Use a JPEG or PNG or WebP image")
    limit = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB")
    return data


@router.post("/probe")
def probe(file: UploadFile = File(...)):
    """Quick pose/quality check of one camera frame (polled while capturing)."""
    data = _read_image(file)
    try:
        r = pipeline_client._post(
            "/probe", files={"file": (file.filename or "frame.jpg", data, file.content_type)}, timeout=15
        )
        return pipeline_client._json_or_raise(r)
    except pipeline_client.NoFaceError as exc:
        raise HTTPException(422, str(exc)) from exc
    except pipeline_client.PipelineUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@router.post("/compare")
def compare(files: list[UploadFile] = File(...)):
    """Similarity matrix between the faces in several photos (are the angles still 'the same
    person' for the recogniser?)."""
    if not 2 <= len(files) <= 12:
        raise HTTPException(422, "Send between 2 and 12 photos")
    payload = [
        ("files", (f.filename or f"shot{i}.jpg", _read_image(f), f.content_type)) for i, f in enumerate(files)
    ]
    try:
        r = pipeline_client._post("/compare", files=payload, timeout=120)  # type: ignore[arg-type]
        return pipeline_client._json_or_raise(r)
    except pipeline_client.NoFaceError as exc:
        raise HTTPException(422, str(exc)) from exc
    except pipeline_client.PipelineUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@router.post("/analyze")
def analyze(file: UploadFile = File(...), box: Optional[str] = Form(None)):
    """Without `box`: every face found in the photo (boxes, landmarks, quality, aligned crop).
    With `box` ("x1,y1,x2,y2" in image pixels): just that face, plus the registered people it
    resembles, so duplicates are flagged *before* anyone is registered."""
    if (file.content_type or "") not in ALLOWED_TYPES:
        raise HTTPException(422, "Unsupported file type. Use a JPEG or PNG or WebP image")
    limit = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB")

    sel: Optional[list[float]] = None
    if box:
        try:
            sel = [float(v) for v in box.split(",")]
            if len(sel) != 4 or sel[2] <= sel[0] or sel[3] <= sel[1]:
                raise ValueError
        except ValueError:
            raise HTTPException(422, "box must be 'x1,y1,x2,y2' with x2>x1 and y2>y1") from None
    try:
        return pipeline_client.analyze_image(data, file.filename or "photo", file.content_type or "", sel)
    except pipeline_client.NoFaceError as exc:
        raise HTTPException(422, str(exc)) from exc
    except pipeline_client.PipelineUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
