"""Face finding in still images (registration / group photos)."""

import base64
from typing import Optional

import cv2
import numpy as np

from .engine import Detection, FaceEngine
from .geometry import iou
from .quality import assess

Box = tuple[float, float, float, float]


def png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def _center_inside(d: Detection, box: Box) -> bool:
    cx, cy = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _select(dets: list[Detection], box: Box) -> Optional[Detection]:
    """The face the user meant by `box`: a face centred inside it (largest wins), else the
    best IoU if it overlaps enough."""
    inside = [d for d in dets if _center_inside(d, box)]
    if inside:
        return max(inside, key=lambda d: d.area)
    best = max(dets, key=lambda d: iou(d.box, box), default=None)
    return best if best is not None and iou(best.box, box) >= 0.3 else None


def _redetect_zoomed(engine: FaceEngine, img: np.ndarray, box: Box) -> Optional[Detection]:
    """The user drew a box where the detector found nothing: zoom into that region
    (small faces often need more pixels) and try again."""
    h, w = img.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    pad = 0.25 * max(bw, bh)
    x1, y1 = max(int(box[0] - pad), 0), max(int(box[1] - pad), 0)
    x2, y2 = min(int(box[2] + pad), w), min(int(box[3] + pad), h)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    crop = img[y1:y2, x1:x2]
    scale = max(1.0, 480.0 / min(crop.shape[:2]))
    up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    found = engine.detect(up)  # live-size detector is fine: the crop is already zoomed
    mapped = [
        Detection(
            box=tuple(float(v / scale + o) for v, o in zip(d.box, (x1, y1, x1, y1))),
            kps=(d.kps / scale) + np.array([x1, y1], dtype=np.float32),
            score=d.score,
        )
        for d in found
    ]
    return _select(mapped, box) if mapped else None


def find_faces(engine: FaceEngine, img: np.ndarray, box: Optional[Box] = None) -> list[Detection]:
    """All faces in the image, or just the one inside `box` (user selection)."""
    dets = engine.detect_multiscale(img)
    if box is None:
        return sorted(dets, key=lambda d: (d.box[1], d.box[0]))  # reading order, stable numbering
    sel = _select(dets, box) or _redetect_zoomed(engine, img, box)
    return [sel] if sel is not None else []


def describe(
    engine: FaceEngine, img: np.ndarray, det: Detection, aligned: Optional[np.ndarray] = None
) -> dict:
    """Everything the UI needs to show about one detected face."""
    if aligned is None:
        aligned = engine.align(img, det)
    return {
        "box": [float(v) for v in det.box],
        "kps": [[float(x), float(y)] for x, y in det.kps],
        "score": float(det.score),
        "quality": assess(aligned, det.box, det.kps, det.score),
        "aligned_png_b64": png_b64(aligned),
    }
