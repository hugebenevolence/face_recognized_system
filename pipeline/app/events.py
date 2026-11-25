"""Saves snapshots and reports recognition events to the backend, off the video thread.

The graph hands over the clean frame as a JPEG it encoded on the GPU (nvjpegenc) and the aligned
face the alignment library produced. Here the frame is cropped around the face for the stored
snapshot, and the evidence coordinates are made relative to that crop.
"""

import io
import logging
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import httpx
import numpy as np
from PIL import Image

from .config import Config
from .picture import png_bytes

log = logging.getLogger("events")
SNAPSHOT_MARGIN = 0.6  # of the face size, on each side


class EventPublisher(threading.Thread):
    def __init__(self, cfg: Config) -> None:
        super().__init__(name="event-publisher", daemon=True)
        self._cfg = cfg
        self._q: queue.Queue = queue.Queue(maxsize=200)
        self._client = httpx.Client(base_url=cfg.backend_url, timeout=10)

    def submit(
        self,
        *,
        status: str,
        person_id: Optional[int],
        confidence: float,
        track_id: int,
        frame_jpeg: bytes,
        box: tuple,
        kps: np.ndarray,
        evidence: Optional[dict] = None,
        aligned: Optional[np.ndarray] = None,
    ) -> None:
        """`frame_jpeg` is the whole clean frame; `box` and `kps` are in frame pixels."""
        item = dict(
            ts=datetime.now(timezone.utc),
            status=status,
            person_id=person_id,
            confidence=confidence,
            track_id=track_id,
            frame_jpeg=frame_jpeg,
            box=box,
            kps=kps,
            evidence=evidence,
            aligned=aligned,
        )
        try:
            self._q.put_nowait(item)
        except queue.Full:
            log.warning("event queue full, dropping event for track %s", track_id)

    def run(self) -> None:
        while True:
            item = self._q.get()
            try:
                self._send(item)
            except Exception:
                log.exception("failed to publish event")

    def _save_snapshot(self, item: dict) -> tuple[Optional[str], tuple[int, int], tuple[int, int]]:
        """Crop of the frame around the face. Returns (path, crop offset, crop size)."""
        frame = Image.open(io.BytesIO(item["frame_jpeg"]))
        w, h = frame.size
        x1, y1, x2, y2 = item["box"]
        mx, my = (x2 - x1) * SNAPSHOT_MARGIN, (y2 - y1) * SNAPSHOT_MARGIN
        a, b = max(int(x1 - mx), 0), max(int(y1 - my), 0)
        c, d = min(int(x2 + mx), w), min(int(y2 + my), h)
        crop = frame.crop((a, b, c, d)).convert("RGB")
        ts = item["ts"]
        rel = f"snapshots/{ts:%Y/%m/%d}/{ts:%H%M%S}_{uuid.uuid4().hex[:8]}.jpg"
        path = self._cfg.data_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(path, format="JPEG", quality=90)
        return rel, (a, b), crop.size

    def _save_aligned(self, snapshot_path: str, rgb: np.ndarray) -> str:
        """112x112 aligned face next to the snapshot (PNG: lossless, so explanations can be
        recomputed later from exactly what the recogniser saw)."""
        rel = snapshot_path.rsplit(".", 1)[0] + "_aligned.png"
        (self._cfg.data_dir / rel).write_bytes(png_bytes(rgb))
        return rel

    def _send(self, item: dict) -> None:
        snapshot_path, (ox, oy), crop_size = self._save_snapshot(item)
        evidence = item.get("evidence")
        if evidence is not None:
            # coordinates relative to the saved snapshot crop, so the UI can draw them on it
            x1, y1, x2, y2 = item["box"]
            evidence["crop_size"] = [int(crop_size[0]), int(crop_size[1])]
            evidence["bbox"] = [float(x1 - ox), float(y1 - oy), float(x2 - ox), float(y2 - oy)]
            evidence["kps"] = [[float(x - ox), float(y - oy)] for x, y in item["kps"]]
            if item.get("aligned") is not None:
                evidence["aligned_path"] = self._save_aligned(snapshot_path, item["aligned"])
        body = {
            "ts": item["ts"].isoformat(),
            "status": item["status"],
            "person_id": item["person_id"],
            "confidence": round(float(item["confidence"]), 4),
            "camera_id": self._cfg.camera_id,
            "track_id": item["track_id"],
            "snapshot_path": snapshot_path,
            "evidence": evidence,
        }
        headers = {"X-Internal-Token": self._cfg.internal_token}
        for attempt in range(1, 4):
            try:
                r = self._client.post("/internal/events", json=body, headers=headers)
                r.raise_for_status()
                return
            except httpx.HTTPError as exc:
                log.warning("publish attempt %d failed: %s", attempt, exc)
                time.sleep(attempt)
        log.error("giving up on event for track %s", item["track_id"])
