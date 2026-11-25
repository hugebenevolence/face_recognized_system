"""Pad probes that turn the DeepStream graph into a face recogniser.

     nvinfer (SCRFD)        nvtracker            nvdspreprocess              nvinfer (ArcFace)      nvdsosd
  ... --> [probe: detect] --> [probe: tracked] --> (face alignment, C++/CUDA) --> [probe: identify] --> boxes

  detect    decodes SCRFD's output tensors into faces (box + five landmarks) and adds them as
            object metadata.
  tracked   pairs each tracked face with its fresh detection and drops tracker-only predictions,
            which have no landmarks. The alignment itself is done by nvdspreprocess
            (pipeline/face_align), which reads SCRFD's landmarks from the same frame's metadata.
  identify  reads the 512-d embedding ArcFace produced, matches it in pgvector, decides
            known / unknown, labels the box and reports events.
  clean     (on the branch that JPEG-encodes clean frames, before drawing) lets through only the
            frames that are needed: a periodic snapshot and the frames of new events. The
            event is sent with that JPEG and the aligned face the alignment library attached.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyds

from .config import Config
from .dsmeta import add_object, iter_list as _iter, read_aligned_face, read_embedding, read_tensors
from .events import EventPublisher
from .geometry import iou as _iou
from .matcher import Matcher
from .quality import assess
from .scrfd import Face, decode

log = logging.getLogger("analytics")
SNAPSHOT_EVERY = 10  # frames between clean snapshots for /snapshot.jpg

GREEN, RED, AMBER = (0.1, 0.75, 0.3), (0.9, 0.2, 0.2), (0.95, 0.65, 0.1)


@dataclass
class TrackState:
    first_frame: int
    hits: int = 0
    last_seen: int = 0
    status: str = "pending"  # pending -> known | unknown
    person_id: Optional[int] = None
    name: Optional[str] = None
    score: float = 0.0
    attempts: int = 0
    last_attempt: int = -(10**9)
    last_event_ts: float = 0.0
    last_event_status: Optional[str] = None
    evidence_top: list = field(default_factory=list)
    face: Optional[Face] = None


class Analytics:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.matcher = Matcher(cfg.database_url)
        self.publisher = EventPublisher(cfg)
        self.explain_overlay = False
        self.frame_idx = 0
        self.last_snapshot: Optional[bytes] = None  # newest clean frame, JPEG encoded on the GPU
        self._wanted: set[int] = set()  # PTS of frames to JPEG-encode on the clean branch
        self._pending: dict[int, list[dict]] = {}  # PTS -> events waiting for that frame's JPEG
        self._states: dict[tuple[int, int], TrackState] = {}
        self._faces: dict[tuple[int, int], list[Face]] = {}  # detections of the frame in flight
        self._aligned: dict[tuple[int, int], dict[int, Face]] = {}  # faces sent to ArcFace, per frame

    def reset(self) -> None:
        """New video source: forget tracks (ids restart) and any half-processed frames."""
        self._states.clear()
        self._faces.clear()
        self._aligned.clear()
        self._wanted.clear()
        self._pending.clear()

    # ------------------------------------------------------------------ probe 1: detect
    def on_detect(self, pad, info, _user=None):
        buf = info.get_buffer()
        bm = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        for fm in _iter(bm.frame_meta_list, pyds.NvDsFrameMeta.cast):
            faces = decode(read_tensors(fm), self.cfg.frame_width, self.cfg.frame_height, self.cfg.det_thresh)
            faces = [f for f in faces if f.size >= self.cfg.min_face_px]
            self._faces[(fm.source_id, fm.frame_num)] = faces
            # nvinfer only sets this for the stock detector output types; nvtracker skips frames
            # without it, and ours is network-type=100 (raw tensors decoded here).
            fm.bInferDone = True
            for f in faces:
                add_object(bm, fm, f.box, f.score, frame_size=(self.cfg.frame_width, self.cfg.frame_height))
        return Gst_PAD_PROBE_OK

    # ------------------------------------------------------------------ probe 2: tracked
    def on_tracked(self, pad, info, _user=None):
        buf = info.get_buffer()
        bm = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        for fm in _iter(bm.frame_meta_list, pyds.NvDsFrameMeta.cast):
            key = (fm.source_id, fm.frame_num)
            faces = self._faces.pop(key, [])
            aligned: dict[int, Face] = {}
            stale = []
            for om in _iter(fm.obj_meta_list, pyds.NvDsObjectMeta.cast):
                r = om.rect_params
                box = (r.left, r.top, r.left + r.width, r.top + r.height)
                best, best_iou = None, 0.4
                for f in faces:
                    v = _iou(box, f.box)
                    if v > best_iou:
                        best, best_iou = f, v
                if best is None:
                    stale.append(om)  # tracker prediction without a fresh detection: no landmarks
                    continue
                aligned[om.object_id] = best
            for om in stale:
                pyds.nvds_remove_obj_meta_from_frame(fm, om)
            self._aligned[key] = aligned
        return Gst_PAD_PROBE_OK

    # ------------------------------------------------------------------ probe 3: identify
    def on_identify(self, pad, info, _user=None):
        buf = info.get_buffer()
        bm = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        for fm in _iter(bm.frame_meta_list, pyds.NvDsFrameMeta.cast):
            self.frame_idx += 1
            self._process_frame(buf, bm, fm)
        return Gst_PAD_PROBE_OK

    def _process_frame(self, buf, bm, fm) -> None:
        cfg = self.cfg
        key = (fm.source_id, fm.frame_num)
        aligned = self._aligned.pop(key, {})
        objs = list(_iter(fm.obj_meta_list, pyds.NvDsObjectMeta.cast))
        if self.frame_idx % SNAPSHOT_EVERY == 0:
            self._wanted.add(buf.pts)  # a clean snapshot for /snapshot.jpg

        seen = set()
        for om in objs:
            face = aligned.get(om.object_id)
            if face is None:
                pyds.nvds_remove_obj_meta_from_frame(fm, om)
                continue
            sid = (fm.source_id, om.object_id)
            seen.add(sid)
            st = self._states.get(sid)
            if st is None:
                st = self._states[sid] = TrackState(first_frame=self.frame_idx)
            st.hits += 1
            st.last_seen = self.frame_idx
            st.face = face

            emb = read_embedding(om)  # absent on frames nvdspreprocess skips (its interval)
            if emb is not None and self.frame_idx - st.last_attempt >= self._interval(st):
                self._identify(st, emb)
            if st.status in ("known", "unknown") and st.hits >= cfg.min_hits:
                self._maybe_emit(st, om, face, buf.pts)
            self._decorate(bm, fm, om, st, face)

        stale = [k for k, s in self._states.items() if self.frame_idx - s.last_seen > 90]
        for k in stale:
            del self._states[k]

    # ------------------------------------------------------------------ decisions
    def _interval(self, st: TrackState) -> int:
        """Frames between database lookups: often while the identity is open, rarely once known."""
        return self.cfg.reid_interval if st.status == "known" else self.cfg.id_interval

    def _identify(self, st: TrackState, emb: np.ndarray) -> None:
        cfg = self.cfg
        st.last_attempt = self.frame_idx
        st.attempts += 1
        tops = self.matcher.top_matches(emb, cfg.model_id, k=3)
        st.evidence_top = [
            {"person_id": m.person_id, "name": m.name, "code": m.code, "score": round(m.score, 4)} for m in tops
        ]
        match = tops[0] if tops else None
        if match and match.score >= cfg.match_threshold:
            st.status, st.score = "known", match.score
            st.person_id, st.name = match.person_id, match.name
        elif st.status != "known":  # a failed re-check never demotes a known track
            st.score = match.score if match else 0.0
            if st.attempts >= cfg.unknown_after:
                st.status, st.person_id, st.name = "unknown", None, None

    def _maybe_emit(self, st: TrackState, om, face: Face, pts: int) -> None:
        now = time.time()
        changed = st.last_event_status != st.status
        if not changed and now - st.last_event_ts < self.cfg.event_cooldown:
            return
        st.last_event_ts, st.last_event_status = now, st.status
        aligned = read_aligned_face(om)  # the 112x112 face ArcFace saw, from the alignment library
        top = st.evidence_top
        evidence = {
            "version": 1,
            "model": self.cfg.model_id,
            "threshold": self.cfg.match_threshold,
            "det_score": float(face.score),
            "quality": assess(aligned, face.box, face.kps, face.score) if aligned is not None else None,
            "top": top,
            "margin": round(top[0]["score"] - top[1]["score"], 4) if len(top) > 1 else None,
            "pipeline": "deepstream",
        }
        self._pending.setdefault(pts, []).append(
            dict(status=st.status, person_id=st.person_id, confidence=st.score, track_id=om.object_id,
                 box=face.box, kps=face.kps.copy(), evidence=evidence, aligned=aligned)
        )
        self._wanted.add(pts)

    # ------------------------------------------------------------------ clean frames
    def clean_gate(self, pad, info, _user=None):
        """Probe on the clean (not yet drawn) branch: only the frames somebody needs go on to
        the JPEG encoder."""
        return Gst_PAD_PROBE_OK if info.get_buffer().pts in self._wanted else Gst_PAD_PROBE_DROP

    def on_clean_jpeg(self, pts: int, jpeg: bytes) -> None:
        """A clean frame, JPEG-encoded on the GPU (nvjpegenc)."""
        self._wanted.discard(pts)
        self.last_snapshot = jpeg
        for item in self._pending.pop(pts, []):
            self.publisher.submit(frame_jpeg=jpeg, **item)
        if len(self._pending) > 32:  # frames that never arrived (e.g. source switched)
            for k in sorted(self._pending)[:-32]:
                del self._pending[k]

    # ------------------------------------------------------------------ drawing
    def _decorate(self, bm, fm, om, st: TrackState, face: Face) -> None:
        if st.status == "known":
            text, color = f"{st.name}  {st.score:.2f}", GREEN
        elif st.status == "unknown":
            text, color = f"Unknown  {st.score:.2f}", RED
        else:
            text, color = "...", AMBER
        r = om.rect_params
        r.border_width = 3
        r.border_color.set(*color, 1.0)
        r.has_bg_color = 0
        tp = om.text_params
        tp.display_text = text
        tp.x_offset = int(max(r.left, 0))
        tp.y_offset = int(max(r.top - 26, 0))
        tp.font_params.font_name = "Serif"
        tp.font_params.font_size = 14
        tp.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        tp.set_bg_clr = 1
        tp.text_bg_clr.set(*color, 0.85)
        if self.explain_overlay:
            self._explain(bm, fm, r, st, face)

    def _explain(self, bm, fm, r, st: TrackState, face: Face) -> None:
        dm = pyds.nvds_acquire_display_meta_from_pool(bm)
        dm.num_circles = 5
        for i, (x, y) in enumerate(face.kps):
            c = dm.circle_params[i]
            c.xc, c.yc, c.radius = int(x), int(y), 3
            c.circle_color.set(1.0, 0.85, 0.1, 1.0)
            c.has_bg_color = 1
            c.bg_color.set(1.0, 0.85, 0.1, 1.0)
        dm.num_labels = 1
        t = dm.text_params[0]
        t.display_text = f"det {face.score:.2f}  sim {st.score:.2f}  thr {self.cfg.match_threshold:.2f}"
        t.x_offset, t.y_offset = int(max(r.left, 0)), int(r.top + r.height + 4)
        t.font_params.font_name = "Serif"
        t.font_params.font_size = 12
        t.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        t.set_bg_clr = 1
        t.text_bg_clr.set(0.0, 0.0, 0.0, 0.7)
        pyds.nvds_add_display_meta_to_frame(fm, dm)


# Gst.PadProbeReturn values without importing Gst here (keeps this module testable)
Gst_PAD_PROBE_DROP = 0
Gst_PAD_PROBE_OK = 1
