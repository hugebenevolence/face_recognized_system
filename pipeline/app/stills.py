"""Face finding in still images (registration / group photos)."""

import base64
from typing import Optional

from .engine import Detection, FaceEngine
from .geometry import iou
from .picture import Picture, png_bytes
from .quality import assess

Box = tuple[float, float, float, float]


def png_b64(rgb) -> str:
    return base64.b64encode(png_bytes(rgb)).decode()


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


def _redetect_zoomed(engine: FaceEngine, pic: Picture, box: Box) -> Optional[Detection]:
    """The user drew a box where the detector found nothing: zoom into that region
    (small faces often need more pixels) and try again. The zoom is done by the graph."""
    bw, bh = box[2] - box[0], box[3] - box[1]
    pad = 0.25 * max(bw, bh)
    x1, y1 = max(box[0] - pad, 0), max(box[1] - pad, 0)
    x2, y2 = min(box[2] + pad, pic.width), min(box[3] + pad, pic.height)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    scale = max(1.0, 480.0 / min(x2 - x1, y2 - y1))
    scale = min(scale, 640.0 / max(x2 - x1, y2 - y1))  # the zoomed region must fit in the view
    found = engine.detect_region(pic, x1, y1, x2, y2, scale)
    return _select(found, box) if found else None


def find_faces(engine: FaceEngine, pic: Picture, box: Optional[Box] = None) -> list[Detection]:
    """All faces in the picture, or just the one inside `box` (user selection)."""
    dets = engine.detect_multiscale(pic)
    if box is None:
        return sorted(dets, key=lambda d: (d.box[1], d.box[0]))  # reading order, stable numbering
    sel = _select(dets, box) or _redetect_zoomed(engine, pic, box)
    return [sel] if sel is not None else []


def describe(engine: FaceEngine, pic: Picture, det: Detection) -> dict:
    """Everything the UI needs to show about one detected face. The aligned face is the one the
    alignment library produced in the graph."""
    aligned = engine.align(pic, det)
    return {
        "box": [float(v) for v in det.box],
        "kps": [[float(x), float(y)] for x, y in det.kps],
        "score": float(det.score),
        "quality": assess(aligned, det.box, det.kps, det.score),
        "aligned_png_b64": png_b64(aligned),
    }
