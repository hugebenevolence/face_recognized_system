"""Thin client for the DeepStream pipeline service's still-image endpoints (registration, group
photos, explanations). The live video is proxied separately in routers/stream.py."""

import json
from typing import Optional

import httpx

from .config import settings


class NoFaceError(ValueError):
    """The image was readable but no usable face was found."""


class PipelineUnavailable(RuntimeError):
    pass


class PipelineConflict(RuntimeError):
    """The pipeline refused (e.g. the event was recorded with a different model)."""


def _box_field(box: Optional[list[float]]) -> dict:
    return {"box": ",".join(f"{v:.1f}" for v in box)} if box else {}


def _post(path: str, *, files: dict, data: Optional[dict] = None, timeout: float = 60) -> httpx.Response:
    try:
        return httpx.post(f"{settings.pipeline_url}{path}", files=files, data=data or {}, timeout=timeout)
    except httpx.HTTPError as exc:
        raise PipelineUnavailable("Recognition service is unreachable") from exc


def _json_or_raise(r: httpx.Response) -> dict:
    if r.status_code == 422:
        raise NoFaceError(r.json().get("detail", "no face found"))
    if r.status_code == 409:
        raise PipelineConflict(r.json().get("detail", "conflict"))
    if r.status_code >= 400:
        raise PipelineUnavailable(f"pipeline error {r.status_code}: {r.text[:200]}")
    return r.json()


def embed_image(data: bytes, filename: str, content_type: str, box: Optional[list[float]] = None) -> dict:
    """Embedding + registration evidence for one face (the one inside `box`, else the largest)."""
    r = _post("/embed", files={"file": (filename, data, content_type)}, data=_box_field(box))
    return _json_or_raise(r)


def analyze_image(data: bytes, filename: str, content_type: str, box: Optional[list[float]] = None) -> dict:
    """All faces in the image, or only the one inside `box` (with who it resembles)."""
    r = _post("/analyze", files={"file": (filename, data, content_type)}, data=_box_field(box), timeout=120)
    return _json_or_raise(r)


def explain(aligned: bytes, crop: Optional[bytes], payload: dict) -> dict:
    files = {"aligned": ("aligned.png", aligned, "image/png")}
    if crop is not None:
        files["crop"] = ("crop.jpg", crop, "image/jpeg")
    # Occlusion analysis runs dozens of model passes; allow time on a slow CPU.
    r = _post("/explain", files=files, data={"payload": json.dumps(payload)}, timeout=300)
    return _json_or_raise(r)


def status() -> dict:
    try:
        return httpx.get(f"{settings.pipeline_url}/status", timeout=5).json()
    except (httpx.HTTPError, ValueError) as exc:
        raise PipelineUnavailable("Recognition service is unreachable") from exc
