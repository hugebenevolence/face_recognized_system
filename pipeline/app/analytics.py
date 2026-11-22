"""Pad probes that turn the DeepStream graph into a face recogniser.

                   nvinfer (SCRFD)        nvtracker        nvinfer (ArcFace)       nvdsosd
   frames ... --> [probe: detect] ---> [probe: roi] ---> [probe: identify] ---> draws the boxes

  detect    decodes SCRFD's output tensors into faces and adds them as object metadata.
  roi       points each tracked face's ROI at the region ArcFace expects (from the landmarks),
            so the secondary nvinfer crops and scales exactly that area on the GPU.
  identify  reads the 512-d embedding nvinfer produced, matches it in pgvector, decides
            known / unknown, labels the box and reports events.
"""

import ctypes
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import pyds

from .config import Config
from .events import EventPublisher
from .matcher import Matcher
from .quality import assess
from .scrfd import Face, decode

log = logging.getLogger("analytics")
_cudart = ctypes.CDLL("libcudart.so.12")
ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

PGIE_ID, SGIE_ID = 1, 2
SKIP_CLASS = 1  # objects given this class are not sent to ArcFace and are dropped before drawing
UNTRACKED_OBJECT_ID = 0xFFFFFFFFFFFFFFFF

# ArcFace reference landmark positions inside the 112x112 face image.
_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float64,
)
_TEMPLATE_MEAN = _TEMPLATE.mean(axis=0)
_TEMPLATE_NORM = float(np.sqrt(((_TEMPLATE - _TEMPLATE_MEAN) ** 2).sum()))

GREEN, RED, AMBER = (0.1, 0.75, 0.3), (0.9, 0.2, 0.2), (0.95, 0.65, 0.1)


def _iter(head, cast):
    while head is not None:
        yield cast(head.data)
        head = head.next


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def face_transform(kps: np.ndarray) -> tuple[float, np.ndarray]:
    """Scale + translation that maps the five landmarks onto the ArcFace template.
    Rotation is left out on purpose: nvinfer can only crop axis-aligned regions, and for
    upright faces the result is within ~1% cosine of a full alignment (checked offline)."""
    mean = kps.mean(axis=0)
    spread = float(np.sqrt(((kps - mean) ** 2).sum())) or 1e-6
    s = _TEMPLATE_NORM / spread
    return s, _TEMPLATE_MEAN - s * mean


def roi_of(kps: np.ndarray) -> tuple[float, float, float]:
    """(left, top, side) of the frame region that becomes the 112x112 ArcFace input."""
    s, t = face_transform(kps)
    return float(-t[0] / s), float(-t[1] / s), 112.0 / s


def aligned_crop(frame_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
    s, t = face_transform(kps)
    m = np.array([[s, 0, t[0]], [0, s, t[1]]], dtype=np.float32)
    return cv2.warpAffine(frame_bgr, m, (112, 112), flags=cv2.INTER_LINEAR)


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
        self.last_raw: Optional[np.ndarray] = None  # newest clean frame (BGR) for /snapshot.jpg
        self._states: dict[tuple[int, int], TrackState] = {}
        self._faces: dict[tuple[int, int], list[Face]] = {}  # detections of the frame in flight
        self._tight: dict[tuple[int, int], dict[int, tuple[Face, tuple]]] = {}

    def reset(self) -> None:
        """New video source: forget tracks (ids restart) and any half-processed frames."""
        self._states.clear()
        self._faces.clear()
        self._tight.clear()

    # ------------------------------------------------------------------ probe 1: detect
    def on_detect(self, pad, info, _user=None):
        buf = info.get_buffer()
        bm = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        for fm in _iter(bm.frame_meta_list, pyds.NvDsFrameMeta.cast):
            tensors = self._read_tensors(fm)
            faces = decode(tensors, self.cfg.frame_width, self.cfg.frame_height, self.cfg.det_thresh)
            faces = [f for f in faces if f.size >= self.cfg.min_face_px]
            self._faces[(fm.source_id, fm.frame_num)] = faces
            # nvinfer only sets this for the stock detector output types; nvtracker skips frames
            # without it, and ours is network-type=100 (raw tensors decoded here).
            fm.bInferDone = True
            for f in faces:
                om = pyds.nvds_acquire_obj_meta_from_pool(bm)
                om.unique_component_id = PGIE_ID
                om.class_id = 0
                om.confidence = f.score
                om.object_id = UNTRACKED_OBJECT_ID
                om.obj_label = "face"
                self._set_rect(om, f.box)
                om.detector_bbox_info.org_bbox_coords.left = f.box[0]
                om.detector_bbox_info.org_bbox_coords.top = f.box[1]
                om.detector_bbox_info.org_bbox_coords.width = f.box[2] - f.box[0]
                om.detector_bbox_info.org_bbox_coords.height = f.box[3] - f.box[1]
                pyds.nvds_add_obj_meta_to_frame(fm, om, None)
        return Gst_PAD_PROBE_OK

    @staticmethod
    def _read_tensors(fm) -> dict[tuple[int, int], np.ndarray]:
        out: dict[tuple[int, int], np.ndarray] = {}
        for um in _iter(fm.frame_user_meta_list, pyds.NvDsUserMeta.cast):
            if um.base_meta.meta_type != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                continue
            tm = pyds.NvDsInferTensorMeta.cast(um.user_meta_data)
            for i in range(tm.num_output_layers):
                layer = pyds.get_nvds_LayerInfo(tm, i)
                total = layer.inferDims.numElements
                rows = layer.inferDims.d[0]
                ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                arr = np.ctypeslib.as_array(ptr, shape=(total,)).reshape(rows, total // rows)
                out[(rows, total // rows)] = arr
        return out

    @staticmethod
    def _set_rect(om, box) -> None:
        r = om.rect_params
        r.left, r.top = box[0], box[1]
        r.width, r.height = box[2] - box[0], box[3] - box[1]

    # ------------------------------------------------------------------ probe 2: roi
    def on_roi(self, pad, info, _user=None):
        buf = info.get_buffer()
        bm = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        for fm in _iter(bm.frame_meta_list, pyds.NvDsFrameMeta.cast):
            key = (fm.source_id, fm.frame_num)
            faces = self._faces.pop(key, [])
            W, H = self.cfg.frame_width, self.cfg.frame_height
            tight: dict[int, tuple[Face, tuple]] = {}
            for om in _iter(fm.obj_meta_list, pyds.NvDsObjectMeta.cast):
                r = om.rect_params
                box = (r.left, r.top, r.left + r.width, r.top + r.height)
                best, best_iou = None, 0.4
                for f in faces:
                    v = _iou(box, f.box)
                    if v > best_iou:
                        best, best_iou = f, v
                if best is None:
                    om.class_id = SKIP_CLASS  # tracker-predicted box without a fresh detection
                    continue
                left, top, side = roi_of(best.kps)
                if left < 0 or top < 0 or left + side > W or top + side > H:
                    om.class_id = SKIP_CLASS  # ROI would leave the frame: not a clean crop
                    tight[om.object_id] = (best, box)
                    continue
                tight[om.object_id] = (best, box)
                st = self._states.get((fm.source_id, om.object_id))
                if st is not None and self.frame_idx - st.last_attempt < self._interval(st):
                    om.class_id = SKIP_CLASS  # identity is still fresh: skip ArcFace for this face
                    continue
                r.left, r.top, r.width, r.height = left, top, side, side
            self._tight[key] = tight
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
        tight = self._tight.pop(key, {})
        objs = list(_iter(fm.obj_meta_list, pyds.NvDsObjectMeta.cast))
        frame = None  # RGBA view of the clean frame, mapped lazily
        # a clean copy for /snapshot.jpg every few frames
        if self.frame_idx % 3 == 0:
            frame = self._map(buf, fm)
            self.last_raw = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        seen = set()
        for om in objs:
            entry = tight.get(om.object_id)
            if entry is None:
                pyds.nvds_remove_obj_meta_from_frame(fm, om)
                continue
            face, box = entry
            self._set_rect(om, box)  # back from the ArcFace ROI to the face box
            om.class_id = 0
            sid = (fm.source_id, om.object_id)
            seen.add(sid)
            st = self._states.get(sid)
            if st is None:
                st = self._states[sid] = TrackState(first_frame=self.frame_idx)
            st.hits += 1
            st.last_seen = self.frame_idx
            st.face = face

            emb = self._embedding(om)  # present only when on_roi let this face through to ArcFace
            if emb is not None:
                self._identify(st, emb)
            if st.status in ("known", "unknown") and st.hits >= cfg.min_hits:
                if frame is None:
                    frame = self._map(buf, fm)
                self._maybe_emit(st, om.object_id, face, frame)
            self._decorate(bm, fm, om, st, face)

        stale = [k for k, s in self._states.items() if self.frame_idx - s.last_seen > 90]
        for k in stale:
            del self._states[k]

    @staticmethod
    def _map(buf, fm) -> np.ndarray:
        """Copy of the RGBA frame in host memory. The pixels stay on the GPU; only the region
        we ask for is copied, with an explicit cudaMemcpy2D (safe next to running kernels)."""
        _, shape, strides, ptr, _ = pyds.get_nvds_buf_surface_gpu(hash(buf), fm.batch_id)
        h, w, c = shape
        out = np.empty((h, w, c), np.uint8)
        err = _cudart.cudaMemcpy2D(
            ctypes.c_void_p(out.ctypes.data), ctypes.c_size_t(w * c),
            ctypes.c_void_p(ctypes.pythonapi.PyCapsule_GetPointer(ptr, None)), ctypes.c_size_t(strides[0]),
            ctypes.c_size_t(w * c), ctypes.c_size_t(h), ctypes.c_int(4),  # 4 = cudaMemcpyDefault
        )
        if err != 0:
            raise RuntimeError(f"cudaMemcpy2D failed with code {err}")
        return out

    @staticmethod
    def _embedding(om) -> Optional[np.ndarray]:
        for um in _iter(om.obj_user_meta_list, pyds.NvDsUserMeta.cast):
            if um.base_meta.meta_type != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                continue
            tm = pyds.NvDsInferTensorMeta.cast(um.user_meta_data)
            if tm.unique_id != SGIE_ID or tm.num_output_layers < 1:
                continue
            layer = pyds.get_nvds_LayerInfo(tm, 0)
            ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
            v = np.ctypeslib.as_array(ptr, shape=(layer.inferDims.numElements,)).astype(np.float32)
            return v / (np.linalg.norm(v) + 1e-9)
        return None

    # ------------------------------------------------------------------ decisions
    def _interval(self, st: TrackState) -> int:
        """Frames between ArcFace runs: often while the identity is open, rarely once known."""
        return self.cfg.reid_interval if st.status == "known" else self.cfg.id_interval

    def _identify(self, st: TrackState, emb: np.ndarray) -> None:
        cfg = self.cfg
        st.last_attempt = self.frame_idx
        st.attempts += 1
        tops = self.matcher.top_matches(emb, cfg.model_pack, k=3)
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

    def _maybe_emit(self, st: TrackState, track_id: int, face: Face, frame_rgba: np.ndarray) -> None:
        now = time.time()
        changed = st.last_event_status != st.status
        if not changed and now - st.last_event_ts < self.cfg.event_cooldown:
            return
        st.last_event_ts, st.last_event_status = now, st.status
        bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
        crop, (ox, oy) = self._crop(bgr, face)
        aligned = aligned_crop(bgr, face.kps)
        top = st.evidence_top
        evidence = {
            "version": 1,
            "model": self.cfg.model_pack,
            "threshold": self.cfg.match_threshold,
            "det_score": float(face.score),
            "crop_size": [int(crop.shape[1]), int(crop.shape[0])],
            # coordinates are relative to the saved snapshot crop, so the UI can draw them on it
            "bbox": [face.box[0] - ox, face.box[1] - oy, face.box[2] - ox, face.box[3] - oy],
            "kps": [[float(x - ox), float(y - oy)] for x, y in face.kps],
            "quality": assess(aligned, face.box, face.kps, face.score),
            "top": top,
            "margin": round(top[0]["score"] - top[1]["score"], 4) if len(top) > 1 else None,
            "pipeline": "deepstream",
        }
        self.publisher.submit(
            status=st.status,
            person_id=st.person_id,
            confidence=st.score,
            track_id=track_id,
            snapshot=crop,
            evidence=evidence,
            aligned=aligned,
        )

    @staticmethod
    def _crop(frame: np.ndarray, face: Face, margin: float = 0.6) -> tuple[np.ndarray, tuple[int, int]]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = face.box
        mx, my = (x2 - x1) * margin, (y2 - y1) * margin
        a, b = max(int(x1 - mx), 0), max(int(y1 - my), 0)
        c, d = min(int(x2 + mx), w), min(int(y2 + my), h)
        return frame[b:d, a:c].copy(), (a, b)

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


# Gst.PadProbeReturn.OK without importing Gst here (keeps this module testable)
Gst_PAD_PROBE_OK = 1
