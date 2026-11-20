import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .db import init_db
from .routers import admin, events, faces, internal, persons, stream
from .ws import manager

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "faces").mkdir(exist_ok=True)
    (settings.data_dir / "snapshots").mkdir(exist_ok=True)
    init_db()
    yield


app = FastAPI(title="Face Recognition API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(persons.router)
app.include_router(faces.router)
app.include_router(admin.router)
app.include_router(events.router)
app.include_router(stream.router)
app.include_router(internal.router)

# Registered face images + event snapshots (paths stored in DB are relative to DATA_DIR).
# StaticFiles needs the directory to exist at import time; lifespan creates it later,
# so make sure it is there now too.
settings.data_dir.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=settings.data_dir), name="files")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.websocket("/ws/events")
async def ws_events(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()  # we only push; this detects disconnects
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(ws)
