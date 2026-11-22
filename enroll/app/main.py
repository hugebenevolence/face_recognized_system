"""Enrollment service: face detection + embedding for still images (registration, group photos,
explanations). The live video is handled by the DeepStream pipeline service."""

import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .config import cfg
from .explain import explain
from .face_engine import FaceEngine
from .matcher import Matcher
from .stills import describe, find_faces

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("enroll")

engine: FaceEngine
matcher: Matcher


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine, matcher
    engine = FaceEngine(cfg)
    matcher = Matcher(cfg.database_url)
    yield


app = FastAPI(title="Face Enrollment", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    return {
        "model": cfg.model_pack,
        "providers": list(cfg.providers),
        "match_threshold": cfg.match_threshold,
    }


# ---------------------------------------------------------------- stills: register / analyse
def _decode(file: UploadFile) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(422, "Not a valid image")
    return img


def _parse_box(box: Optional[str]) -> Optional[tuple]:
    if not box:
        return None
    try:
        v = [float(x) for x in box.split(",")]
        if len(v) != 4 or v[2] <= v[0] or v[3] <= v[1]:
            raise ValueError
    except ValueError:
        raise HTTPException(422, "box must be 'x1,y1,x2,y2' in image pixels") from None
    return tuple(v)


def _similar(emb: np.ndarray) -> list[dict]:
    return [
        {"person_id": m.person_id, "name": m.name, "code": m.code, "score": round(m.score, 4)}
        for m in matcher.top_matches(emb, engine.pack, k=3)
    ]


@app.post("/embed")
def embed(file: UploadFile = File(...), box: Optional[str] = Form(None)):
    """Embedding of one face in an uploaded image: the one inside `box` if given, otherwise
    the largest. Also returns the evidence stored with the registration."""
    img = _decode(file)
    b = _parse_box(box)
    dets = find_faces(engine, img, b)
    if not dets:
        raise HTTPException(422, "No face found in the selected area" if b else "No face detected")
    best = max(dets, key=lambda d: d.area)
    if best.score < 0.5:
        raise HTTPException(422, f"Detection confidence is too low at {best.score:.2f}")
    if best.size < 60:
        raise HTTPException(422, "Face is too small. It needs at least 60 px")
    aligned = engine.align(img, best)
    emb = engine.embed_aligned(aligned)
    info = describe(engine, img, best, aligned)
    return {
        "model": engine.pack,
        "embedding": emb.tolist(),
        "det_score": best.score,
        "face_count": len(dets),
        "image_size": [int(img.shape[1]), int(img.shape[0])],
        "bbox": info["box"],
        **info,
    }


@app.post("/analyze")
def analyze(file: UploadFile = File(...), box: Optional[str] = Form(None)):
    """All faces in an image (for the group-photo picker), or only the one inside `box`,
    in which case it also reports which registered people it resembles."""
    img = _decode(file)
    b = _parse_box(box)
    dets = find_faces(engine, img, b)
    if b is not None and not dets:
        raise HTTPException(422, "No face found in the selected area")
    faces = []
    for i, d in enumerate(dets[:MAX_FACES]):
        f = describe(engine, img, d)
        f["index"] = i
        faces.append(f)
    if b is not None:
        faces[0]["similar"] = _similar(engine.embed(img, dets[0]))
    return {
        "image": {"width": int(img.shape[1]), "height": int(img.shape[0])},
        "model": engine.pack,
        "faces": faces,
        "truncated": len(dets) > MAX_FACES,
    }


MAX_FACES = 60


@app.post("/probe")
def probe(file: UploadFile = File(...)):
    """Cheap per-frame check used while capturing from a camera: is there exactly one
    usable face and how is it posed? (live detector size, no embedding, no PNGs)"""
    img = _decode(file)
    dets = engine.detect(img)
    if not dets:
        return {"faces": 0}
    best = max(dets, key=lambda d: d.area)
    aligned = engine.align(img, best)
    d = describe(engine, img, best, aligned)
    return {
        "faces": len(dets),
        "box": d["box"],
        "score": d["score"],
        "quality": d["quality"],
        "image_size": [int(img.shape[1]), int(img.shape[0])],
    }


@app.post("/compare")
def compare(files: list[UploadFile] = File(...)):
    """Pairwise cosine similarity between the largest face of each image. Used to show how
    much a person's appearance changes with head angle, i.e. whether extra angles are needed."""
    embs: list[Optional[np.ndarray]] = []
    items: list[dict] = []
    for f in files[:12]:
        img = _decode(f)
        dets = find_faces(engine, img, None)
        if not dets:
            embs.append(None)
            items.append({"ok": False})
            continue
        best = max(dets, key=lambda d: d.area)
        aligned = engine.align(img, best)
        embs.append(engine.embed_aligned(aligned))
        q = describe(engine, img, best, aligned)["quality"]
        items.append(
            {"ok": True, "yaw_ratio": q["metrics"]["yaw_ratio"], "quality": q["overall"], "faces": len(dets)}
        )
    matrix = [[None if a is None or b is None else round(float(a @ b), 4) for b in embs] for a in embs]
    return {"model": engine.pack, "threshold": cfg.match_threshold, "items": items, "matrix": matrix}


# ---------------------------------------------------------------- explainability
@app.post("/explain")
def explain_face(
    aligned: UploadFile = File(...),
    payload: str = Form(...),
    crop: Optional[UploadFile] = File(None),
):
    """Occlusion-sensitivity maps. `payload`: {"refs": [[512 floats]...], "bbox": [...], "kps": [[x,y]*5]}
    where bbox/kps are relative to `crop`. Heavy (tens of model passes): call on demand."""
    try:
        p = json.loads(payload)
    except ValueError:
        raise HTTPException(422, "payload must be JSON") from None
    if p.get("model") and p["model"] != engine.pack:
        raise HTTPException(
            409,
            f"This event was recorded with model {p['model']} but the system now runs {engine.pack} "
            "so it cannot be explained",
        )
    a = _decode(aligned)
    if a.shape[:2] != (112, 112):
        a = cv2.resize(a, (112, 112), interpolation=cv2.INTER_AREA)
    refs = np.asarray(p.get("refs") or [], dtype=np.float32).reshape(-1, 512)
    refs = refs / (np.linalg.norm(refs, axis=1, keepdims=True) + 1e-9) if len(refs) else refs
    c = _decode(crop) if crop is not None else None
    if c is not None and not (p.get("bbox") and p.get("kps")):
        c = None
    return explain(engine, a, c, p.get("bbox"), p.get("kps"), refs)
