"""Proxies the pipeline's video + status so the browser only talks to the backend."""

import logging

import httpx
import websockets
from fastapi import APIRouter, Body, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from ..config import settings

log = logging.getLogger("stream")
# Prefix is /api for REST; the WebSocket route below is mounted under it as /api/ws/camera.
router = APIRouter(prefix="/api", tags=["stream"])


@router.get("/stream.mjpg")
async def mjpeg_stream():
    client = httpx.AsyncClient(timeout=httpx.Timeout(10, read=None))
    try:
        req = client.build_request("GET", f"{settings.pipeline_url}/stream.mjpg")
        upstream = await client.send(req, stream=True)
    except httpx.HTTPError as exc:
        await client.aclose()
        raise HTTPException(503, "Recognition service is unreachable") from exc

    async def body():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        body(),
        media_type=upstream.headers.get("content-type", "multipart/x-mixed-replace"),
        headers={"Cache-Control": "no-store"},
    )


@router.get("/snapshot.jpg")
async def snapshot():
    """Latest un-annotated frame (used by 'capture from stream' when registering)."""
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.get(f"{settings.pipeline_url}/snapshot.jpg")
        except httpx.HTTPError as exc:
            raise HTTPException(503, "Recognition service is unreachable") from exc
    if r.status_code != 200:
        raise HTTPException(r.status_code, "no frame available yet")
    return Response(r.content, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/sources")
async def sources():
    """Server-side video sources the UI can switch to (presets + files in /samples)."""
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            r = await client.get(f"{settings.pipeline_url}/sources")
        except httpx.HTTPError as exc:
            raise HTTPException(503, "Recognition service is unreachable") from exc
    return Response(r.content, status_code=r.status_code, media_type="application/json")


@router.put("/source")
async def set_source(body: dict = Body(...)):
    """Switch what the pipeline analyses: {"kind":"uri","uri":"rtsp://..."} or {"kind":"browser"}."""
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.put(f"{settings.pipeline_url}/source", json=body)
        except httpx.HTTPError as exc:
            raise HTTPException(503, "Recognition service is unreachable") from exc
    return Response(r.content, status_code=r.status_code, media_type="application/json")


@router.websocket("/ws/camera")
async def camera_relay(ws: WebSocket):
    """Browser camera -> pipeline. The page sends binary JPEG frames; we forward them."""
    await ws.accept()
    upstream_url = settings.pipeline_url.replace("http", "ws", 1) + "/ingest"
    try:
        async with websockets.connect(upstream_url, max_size=4 * 1024 * 1024) as upstream:
            while True:
                await upstream.send(await ws.receive_bytes())
    except (WebSocketDisconnect, websockets.ConnectionClosed, KeyError, RuntimeError):
        pass
    except OSError as exc:  # pipeline down
        log.warning("camera relay: pipeline unreachable: %s", exc)
    finally:
        try:
            await ws.close()
        except RuntimeError:
            pass


@router.get("/overlay")
async def get_overlay():
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            r = await client.get(f"{settings.pipeline_url}/overlay")
        except httpx.HTTPError as exc:
            raise HTTPException(503, "Recognition service is unreachable") from exc
    return Response(r.content, status_code=r.status_code, media_type="application/json")


@router.put("/overlay")
async def set_overlay(body: dict = Body(...)):
    """Turn the explain overlay (landmarks + score details) on the live video on or off."""
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            r = await client.put(f"{settings.pipeline_url}/overlay", json=body)
        except httpx.HTTPError as exc:
            raise HTTPException(503, "Recognition service is unreachable") from exc
    return Response(r.content, status_code=r.status_code, media_type="application/json")


@router.get("/status")
async def status():
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            r = await client.get(f"{settings.pipeline_url}/status")
            return r.json()
        except httpx.HTTPError:
            return {"connected": False, "pipeline_up": False, "camera_id": settings.camera_id}
