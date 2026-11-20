"""GStreamer capture + per-frame detect / track / recognise / annotate loop."""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from .config import Config  # noqa: E402
from .draw import AMBER, GREEN, RED, draw_label, draw_landmarks, draw_note, placeholder  # noqa: E402
from .events import EventPublisher  # noqa: E402
from .face_engine import Detection, FaceEngine  # noqa: E402
from .matcher import Matcher  # noqa: E402
from .quality import assess  # noqa: E402
from .tracker import IoUTracker, Track  # noqa: E402

log = logging.getLogger("runner")
Gst.init(None)


@dataclass(frozen=True)
class SourceSpec:
    kind: str  # "uri": GStreamer reads it | "browser": JPEG frames pushed over WebSocket
    uri: str = ""


def build_pipeline_desc(source: str) -> tuple[str, bool]:
    """Returns (gst-launch description, is_live). Files are paced in real time."""
    if source.startswith(("rtsp://", "rtsps://")):
        head, live = f'rtspsrc location="{source}" latency=200 protocols=tcp ! decodebin', True
    elif source.startswith("v4l2:"):
        head, live = f"v4l2src device={source[5:]} ! decodebin", True
    else:
        path = source[7:] if source.startswith("file://") else source
        head, live = f'filesrc location="{path}" ! decodebin', False
    sink = (
        "videoconvert ! video/x-raw,format=BGR ! "
        f"appsink name=sink max-buffers=1 drop=true sync={'false' if live else 'true'}"
    )
    return f"{head} ! {sink}", live


class Runner(threading.Thread):
    def __init__(self, cfg: Config, engine: FaceEngine) -> None:
        super().__init__(name="runner", daemon=True)
        self.cfg = cfg
        self.engine = engine
        self.matcher = Matcher(cfg.database_url)
        self.publisher = EventPublisher(cfg)
        self.tracker = IoUTracker(cfg.iou_threshold, cfg.max_age)

        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._frame_idx = 0
        self._jpeg: bytes = self._encode(placeholder("Waiting for video stream"))
        self._jpeg_seq = 0
        self._raw: Optional[np.ndarray] = None
        self._times: deque[float] = deque(maxlen=30)
        self.connected = False
        self.last_error: Optional[str] = None
        self.explain_overlay = False  # draw landmarks + score details on the video

        # Runtime source switching. `_gen` bumps on every switch so the running capture
        # loop notices and exits; `_switch` also wakes the supervisor's backoff sleep.
        self._spec = SourceSpec("uri", cfg.source_uri)
        self._gen = 0
        self._switch = threading.Event()
        # Browser camera: only the newest frame matters, older ones are dropped.
        self._browser_cv = threading.Condition()
        self._browser_frame: Optional[bytes] = None
        self.browser_clients = 0

    # ---------- source selection ----------
    @property
    def source(self) -> SourceSpec:
        return self._spec

    def set_source(self, spec: SourceSpec) -> None:
        with self._lock:
            self._spec = spec
            self._gen += 1
            self.last_error = None
        with self._browser_cv:
            self._browser_frame = None
            self._browser_cv.notify_all()
        self._switch.set()
        log.info("source switched to %s", spec)

    def push_jpeg(self, data: bytes) -> None:
        """Called by the /ingest WebSocket; ignored unless the browser source is active."""
        if self._spec.kind != "browser":
            return
        with self._browser_cv:
            self._browser_frame = data
            self._browser_cv.notify()

    # ---------- public accessors (called from the API thread) ----------
    def latest_jpeg(self) -> tuple[bytes, int]:
        with self._lock:
            return self._jpeg, self._jpeg_seq

    def latest_raw_jpeg(self) -> Optional[bytes]:
        with self._lock:
            raw = self._raw
        return self._encode(raw) if raw is not None else None

    @property
    def fps(self) -> float:
        t = list(self._times)
        if len(t) < 2 or t[-1] <= t[0] or time.monotonic() - t[-1] > 3:  # stale = stalled stream
            return 0.0
        return round((len(t) - 1) / (t[-1] - t[0]), 1)

    def stop(self) -> None:
        self._stop_evt.set()
        self._switch.set()  # wake the supervisor's backoff sleep

    # ---------- thread body ----------
    def run(self) -> None:
        self.publisher.start()
        while not self._stop_evt.is_set():
            self._switch.clear()  # clear *before* reading, so a concurrent switch isn't lost
            with self._lock:
                spec, gen = self._spec, self._gen
            self.tracker = IoUTracker(self.cfg.iou_threshold, self.cfg.max_age)  # fresh ids
            try:
                if spec.kind == "browser":
                    self._browser_loop(gen)
                else:
                    self._capture_once(spec.uri, gen)
            except Exception as exc:  # keep the supervisor alive no matter what
                self.last_error = str(exc)
                log.exception("capture loop crashed")
            self.connected = False
            msg = "Waiting for browser camera" if spec.kind == "browser" else "Reconnecting to video source"
            self._publish(placeholder(msg), raw=None)
            self._switch.wait(3)  # backoff (also what loops a file at EOS); a switch cuts it short

    def _browser_loop(self, gen: int) -> None:
        last_frame = time.monotonic()
        while not self._stop_evt.is_set() and self._gen == gen:
            with self._browser_cv:
                if self._browser_frame is None:
                    self._browser_cv.wait(1.0)
                data, self._browser_frame = self._browser_frame, None
            if data is None:
                if self.connected and time.monotonic() - last_frame > 3:
                    self.connected = False  # browser tab stopped sending
                    self._publish(placeholder("Waiting for browser camera"), raw=None)
                continue
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            last_frame = time.monotonic()
            self.connected, self.last_error = True, None
            self._handle_frame(frame)

    def _capture_once(self, uri: str, gen: int) -> None:
        desc, live = build_pipeline_desc(uri)
        log.info("starting pipeline: %s", desc)
        pipe = Gst.parse_launch(desc)
        sink = pipe.get_by_name("sink")
        bus = pipe.get_bus()
        pipe.set_state(Gst.State.PLAYING)
        last_frame = time.monotonic()
        try:
            while not self._stop_evt.is_set() and self._gen == gen:
                sample = sink.emit("try-pull-sample", Gst.SECOND)
                if sample is None:
                    msg = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
                    if msg is not None:
                        if msg.type == Gst.MessageType.ERROR:
                            err, dbg = msg.parse_error()
                            self.last_error = f"{err.message}"
                            log.error("gst error: %s (%s)", err.message, dbg)
                        else:
                            log.info("end of stream")
                        return
                    if time.monotonic() - last_frame > 15:
                        self.last_error = "no frames for 15s"
                        log.error(self.last_error)
                        return
                    continue
                frame = self._sample_to_bgr(sample)
                if frame is None:
                    continue
                last_frame = time.monotonic()
                self.connected, self.last_error = True, None
                self._handle_frame(frame)
        finally:
            pipe.set_state(Gst.State.NULL)

    @staticmethod
    def _sample_to_bgr(sample) -> Optional[np.ndarray]:
        caps = sample.get_caps().get_structure(0)
        w, h = caps.get_value("width"), caps.get_value("height")
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None
        try:
            stride = (w * 3 + 3) // 4 * 4  # GStreamer pads rows to 4 bytes
            data = np.frombuffer(info.data, dtype=np.uint8)[: stride * h]
            return data.reshape(h, stride)[:, : w * 3].reshape(h, w, 3).copy()
        finally:
            buf.unmap(info)

    # ---------- per-frame logic ----------
    def _handle_frame(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if w > self.cfg.max_width:
            s = self.cfg.max_width / w
            frame = cv2.resize(frame, (self.cfg.max_width, int(h * s)), interpolation=cv2.INTER_AREA)

        self._frame_idx += 1
        self._times.append(time.monotonic())
        cfg = self.cfg

        dets = [
            d for d in self.engine.detect(frame) if d.score >= cfg.det_thresh and d.size >= cfg.min_face_px
        ]
        tracks = self.tracker.update([d.box for d in dets])
        now = time.time()

        clean = frame.copy()  # un-annotated copy for snapshots / registration capture
        explain = self.explain_overlay
        for det, tr in zip(dets, tracks):
            self._identify(clean, det, tr)
            self._maybe_emit(clean, det, tr, now)
            draw_label(frame, det.box, *self._label(tr))
            if explain:
                draw_landmarks(frame, det.kps)
                draw_note(frame, det.box, self._detail(det, tr))
        self._publish(frame, raw=clean)

    def _detail(self, det: Detection, tr: Track) -> str:
        thr = self.cfg.match_threshold
        return f"det {det.score:.2f}  sim {tr.score:.2f}  thr {thr:.2f}"

    def _identify(self, frame: np.ndarray, det: Detection, tr: Track) -> None:
        cfg = self.cfg
        interval = cfg.reid_interval if tr.status == "known" else cfg.id_interval
        if self._frame_idx - tr.last_attempt_frame < interval:
            return
        tr.last_attempt_frame = self._frame_idx
        tr.attempts += 1

        aligned = self.engine.align(frame, det)
        tops = self.matcher.top_matches(self.engine.embed_aligned(aligned), self.engine.pack, k=3)
        match = tops[0] if tops else None
        # Kept on the track so an event can later show *why* it was decided this way.
        tr.extra["aligned"] = aligned
        tr.extra["quality"] = assess(aligned, det.box, det.kps, det.score)
        tr.extra["top"] = [
            {"person_id": m.person_id, "name": m.name, "code": m.code, "score": round(m.score, 4)}
            for m in tops
        ]
        if match and match.score >= cfg.match_threshold:
            tr.status, tr.score = "known", match.score
            tr.person_id, tr.name, tr.code = match.person_id, match.name, match.code
        elif tr.status != "known":  # a failed re-check never demotes a known track
            tr.score = match.score if match else 0.0
            if tr.attempts >= cfg.unknown_after:
                tr.status, tr.person_id, tr.name, tr.code = "unknown", None, None, None

    def _maybe_emit(self, frame: np.ndarray, det: Detection, tr: Track, now: float) -> None:
        if tr.hits < self.cfg.min_hits or tr.status not in ("known", "unknown"):
            return
        changed = tr.last_event_status != tr.status
        if not changed and now - tr.last_event_ts < self.cfg.event_cooldown:
            return
        tr.last_event_ts, tr.last_event_status = now, tr.status
        crop, (ox, oy) = self._crop(frame, det)
        aligned = tr.extra.get("aligned")
        evidence = None
        if aligned is not None:
            top = tr.extra.get("top", [])
            evidence = {
                "version": 1,
                "model": self.engine.pack,
                "threshold": self.cfg.match_threshold,
                "det_score": float(det.score),
                "crop_size": [int(crop.shape[1]), int(crop.shape[0])],
                # coordinates are relative to the saved snapshot crop, so the UI can draw them on it
                "bbox": [det.box[0] - ox, det.box[1] - oy, det.box[2] - ox, det.box[3] - oy],
                "kps": [[float(x - ox), float(y - oy)] for x, y in det.kps],
                "quality": tr.extra.get("quality"),
                "top": top,
                "margin": round(top[0]["score"] - top[1]["score"], 4) if len(top) > 1 else None,
            }
        self.publisher.submit(
            status=tr.status,
            person_id=tr.person_id,
            confidence=tr.score,
            track_id=tr.id,
            snapshot=crop,
            evidence=evidence,
            aligned=aligned,
        )

    @staticmethod
    def _crop(frame: np.ndarray, det: Detection, margin: float = 0.6) -> tuple[np.ndarray, tuple[int, int]]:
        """Padded face crop and its (x, y) offset inside the frame."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = det.box
        mx, my = (x2 - x1) * margin, (y2 - y1) * margin
        a, b = max(int(x1 - mx), 0), max(int(y1 - my), 0)
        c, d = min(int(x2 + mx), w), min(int(y2 + my), h)
        return frame[b:d, a:c].copy(), (a, b)

    @staticmethod
    def _label(tr: Track) -> tuple[str, tuple[int, int, int]]:
        if tr.status == "known":
            return f"{tr.name}  {tr.score:.2f}", GREEN
        if tr.status == "unknown":
            return f"Unknown  {tr.score:.2f}", RED
        return "...", AMBER

    # ---------- output ----------
    def _encode(self, img: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
        return buf.tobytes() if ok else b""

    def _publish(self, annotated: np.ndarray, raw: Optional[np.ndarray]) -> None:
        jpeg = self._encode(annotated)
        with self._lock:
            self._jpeg, self._jpeg_seq = jpeg, self._jpeg_seq + 1
            self._raw = raw
