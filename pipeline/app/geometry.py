"""Box and landmark geometry shared by the live and still-image graphs."""

import numpy as np

# ArcFace reference landmark positions inside the 112x112 face image:
# left eye, right eye, nose tip, left mouth corner, right mouth corner.
TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float64,
)


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


def alignment_matrix(kps: np.ndarray, dst: np.ndarray = TEMPLATE) -> np.ndarray:
    """Face alignment: the 2x3 similarity transform (rotation, uniform scale, translation) that
    maps the five detected landmarks onto the ArcFace template, by least squares (Umeyama).
    This is the transform insightface's norm_crop uses."""
    src = np.asarray(kps, dtype=np.float64).reshape(-1, 2)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    u, sv, vt = np.linalg.svd(cov)
    d = np.array([1.0, -1.0 if np.linalg.det(u) * np.linalg.det(vt) < 0 else 1.0])
    rot = u @ np.diag(d) @ vt
    var = (sc**2).sum() / len(src)
    scale = float((sv * d).sum() / var) if var > 0 else 1.0
    t = mu_d - scale * rot @ mu_s
    return np.hstack([scale * rot, t[:, None]])


def face_extent(kps: np.ndarray) -> float:
    """Side, in image pixels, of the region that becomes the 112x112 aligned face."""
    m = alignment_matrix(kps)
    return 112.0 / float(np.hypot(m[0, 0], m[1, 0]))
