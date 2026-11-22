"""Pipeline service: runs the DeepStream graph and serves its video + status over HTTP."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from .config import cfg
from .runner import Runner, SourceSpec
from .sources import list_file_sources, public_uri, validate_uri

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

runner: Runner
BOUNDARY = b"frame"
MAX_INGEST_FRAME = 2 * 1024 * 1024


@asynccontextmanager
async def lifespan(_: FastAPI):
    global runner
    runner = Runner(cfg)
    runner.start()
    yield
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
        "model": cfg.model_pack,
        "engine": "DeepStream / TensorRT",
        "match_threshold": cfg.match_threshold,
        "explain_overlay": runner.explain_overlay,
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
