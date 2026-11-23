"""Box and landmark geometry shared by the live probes and the still-image service."""

import cv2
import numpy as np

# ArcFace reference landmark positions inside the 112x112 face image:
# left eye, right eye, nose tip, left mouth corner, right mouth corner.
TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float64,
)
_TEMPLATE_MEAN = TEMPLATE.mean(axis=0)
_TEMPLATE_NORM = float(np.sqrt(((TEMPLATE - _TEMPLATE_MEAN) ** 2).sum()))


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def containment(a, b) -> float:
    """Share of the smaller box that lies inside the other one."""
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / small if small > 0 else 0.0


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
    """The 112x112 face exactly as ArcFace sees it."""
    s, t = face_transform(kps)
    m = np.array([[s, 0, t[0]], [0, s, t[1]]], dtype=np.float32)
    return cv2.warpAffine(frame_bgr, m, (112, 112), flags=cv2.INTER_LINEAR)
