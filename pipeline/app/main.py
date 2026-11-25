"""Pipeline service: runs the DeepStream graphs and serves their video, status and still-image
analysis over HTTP.

  live graph   video source -> detect -> track -> embed -> draw      (runner.py, analytics.py)
  still graph  one photo at a time -> detect / embed                 (still_graph.py, engine.py)

Both graphs use the same two TensorRT models, so a face registered from a photo and the same
face seen on camera are embedded by the same code.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Literal, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from .config import cfg
from .engine import FaceEngine
from .explain import explain
from .runner import Runner, SourceSpec
from .sources import list_file_sources, public_uri, validate_uri
from .still_graph import GraphNotReady, StillGraph
from .stills import describe, find_faces

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

runner: Runner
engine: FaceEngine
BOUNDARY = b"frame"
MAX_INGEST_FRAME = 2 * 1024 * 1024
MAX_FACES = 60


@asynccontextmanager
async def lifespan(_: FastAPI):
    global runner, engine
    runner = Runner(cfg)
    runner.start()
    graph = StillGraph(cfg, runner.engines_ready)
    graph.start()
    engine = FaceEngine(cfg, graph)
    yield
    graph.stop()
    runner.stop()


app = FastAPI(title="Face Pipeline (DeepStream)", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    src = runner.source
    return {
        "pipeline_up": True,
        "camera_id": cfg.camera_id,
        "source_kind": src.kind,
        "source": "browser camera" if src.kind == "browser" else public_uri(src.uri),
        "browser_clients": runner.browser_clients,
        "connected": runner.connected,
        "fps": runner.fps,
        "error": runner.last_error,
        "model": cfg.model_id,
        "engine": "DeepStream / TensorRT",
        "match_threshold": cfg.match_threshold,
        "explain_overlay": runner.explain_overlay,
        "stills_ready": engine.ready,
    }


class SourceIn(BaseModel):
    kind: Literal["uri", "browser"]
    uri: Optional[str] = None


@app.get("/sources")
def sources():
    src = runner.source
    return {
        "presets": [dict(p) for p in cfg.source_presets] + list_file_sources(),
        "current": {"kind": src.kind, "uri": None if src.kind == "browser" else public_uri(src.uri)},
    }


@app.put("/source")
def set_source(body: SourceIn):
    if body.kind == "browser":
        runner.set_source(SourceSpec("browser"))
    else:
        try:
            uri = validate_uri(body.uri or "")
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        runner.set_source(SourceSpec("uri", uri))
    return sources()["current"]


@app.websocket("/ingest")
async def ingest(ws: WebSocket):
    """Receives JPEG frames from a browser camera (relayed by the backend)."""
    await ws.accept()
    runner.browser_clients += 1
    try:
        while True:
            data = await ws.receive_bytes()
            if len(data) <= MAX_INGEST_FRAME:
                runner.push_jpeg(data)
    except (WebSocketDisconnect, KeyError, RuntimeError):
        pass
    finally:
        runner.browser_clients -= 1


@app.get("/stream.mjpg")
async def stream():
    async def frames():
        last = -1
        while True:
            jpeg, seq = runner.latest_jpeg()
            if seq != last:
                last = seq
                yield (
                    b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
                )
            await asyncio.sleep(0.02)

    return StreamingResponse(
        frames(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/snapshot.jpg")
def snapshot():
    jpeg = runner.latest_raw_jpeg()
    if jpeg is None:
        raise HTTPException(404, "no frame yet")
    return Response(jpeg, media_type="image/jpeg")


class OverlayIn(BaseModel):
    explain: bool


@app.get("/overlay")
def get_overlay():
    return {"explain": runner.explain_overlay}


@app.put("/overlay")
def set_overlay(body: OverlayIn):
    runner.explain_overlay = body.explain
    return {"explain": runner.explain_overlay}


# ---------------------------------------------------------------- stills: register / analyse
@app.exception_handler(GraphNotReady)
async def _not_ready(_request, exc: GraphNotReady):
    return JSONResponse({"detail": str(exc)}, status_code=503)


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
        for m in runner.analytics.matcher.top_matches(emb, cfg.model_id, k=3)
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
    emb = engine.embed(img, best)
    info = describe(engine, img, best, aligned)
    return {
        "model": engine.pack,
        "embedding": emb.tolist(),
        "det_score": best.score,
        "face_count": len(dets),
        "image_size": [int(img.shape[1]), int(img.shape[0])],
        "bbox": info["box"],
        "pipeline": "deepstream",
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


@app.post("/probe")
def probe(file: UploadFile = File(...)):
    """Cheap per-frame check used while capturing from a camera: is there exactly one
    usable face and how is it posed? (detection only, no embedding)"""
    img = _decode(file)
    dets = engine.detect(img)
    if not dets:
        return {"faces": 0}
    best = max(dets, key=lambda d: d.area)
    d = describe(engine, img, best, engine.align(img, best))
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
        embs.append(engine.embed(img, best))
        q = describe(engine, img, best, engine.align(img, best))["quality"]
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
    where bbox/kps are relative to `crop`. Heavy (tens of passes through the graph): on demand only."""
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
