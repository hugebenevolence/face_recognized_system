"""Saves snapshots and reports recognition events to the backend, off the video thread."""

import logging
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import cv2
import httpx
import numpy as np

from .config import Config

log = logging.getLogger("events")


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
        snapshot: np.ndarray,
        evidence: Optional[dict] = None,
        aligned: Optional[np.ndarray] = None,
    ) -> None:
        item = dict(
            ts=datetime.now(timezone.utc),
            status=status,
            person_id=person_id,
            confidence=confidence,
            track_id=track_id,
            snapshot=snapshot,
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

    def _save_snapshot(self, ts: datetime, img: np.ndarray) -> Optional[str]:
        rel = f"snapshots/{ts:%Y/%m/%d}/{ts:%H%M%S}_{uuid.uuid4().hex[:8]}.jpg"
        path = self._cfg.data_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return rel if ok else None

    def _save_aligned(self, snapshot_path: Optional[str], img: np.ndarray) -> Optional[str]:
        """112x112 aligned face next to the snapshot (PNG: lossless, so explanations can be
        recomputed later from exactly what the recogniser saw)."""
        if snapshot_path is None:
            return None
        rel = snapshot_path.rsplit(".", 1)[0] + "_aligned.png"
        return rel if cv2.imwrite(str(self._cfg.data_dir / rel), img) else None

    def _send(self, item: dict) -> None:
        snapshot_path = self._save_snapshot(item["ts"], item["snapshot"])
        evidence = item.get("evidence")
        if evidence is not None and item.get("aligned") is not None:
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
