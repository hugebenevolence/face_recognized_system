"""Occlusion-sensitivity explanations.

For a given face we hide one patch at a time and measure how much a score drops:
  * recognition: cosine similarity to the matched person's stored face (on the aligned 112x112)
  * detection:   the detector's confidence for that face (on the stored face crop)
Regions whose occlusion hurts the score most are the ones the model relies on.

This describes the model's sensitivity, not human-style reasoning, and the result depends
on patch size and fill colour. It is computed on demand from stored crops, never per frame.
"""

import base64
import threading
import time

import cv2
import numpy as np

from .face_engine import FaceEngine
from .geometry import iou

# Landmark layout of an aligned ArcFace crop (insightface's fixed 112x112 template):
# left eye, right eye, nose tip, left mouth corner, right mouth corner.
ALIGNED_KPS = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)
REGION_ORDER = ["Eyes", "Nose", "Mouth", "Forehead", "Cheeks and jaw"]

# Heavy: one at a time so an explanation never starves the live loop.
_gate = threading.Semaphore(1)


def _png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def _region_masks(kps: np.ndarray, shape: tuple, face_mask: np.ndarray) -> dict[str, np.ndarray]:
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    le, re_, nose, ml, mr = (np.asarray(k, dtype=np.float32) for k in kps)
    iod = float(np.linalg.norm(re_ - le))
    mouth = (ml + mr) / 2
    eye_y = (le[1] + re_[1]) / 2

    def near(c: np.ndarray, r: float) -> np.ndarray:
        return (xx - c[0]) ** 2 + (yy - c[1]) ** 2 <= r * r

    raw = {
        "Eyes": near(le, 0.45 * iod) | near(re_, 0.45 * iod),
        "Nose": near(nose, 0.40 * iod),
        "Mouth": near(mouth, 0.55 * iod),
        "Forehead": yy < eye_y - 0.45 * iod,
    }
    taken = np.zeros((h, w), bool)
    out: dict[str, np.ndarray] = {}
    for name in REGION_ORDER[:-1]:
        m = raw[name] & face_mask & ~taken  # first region wins on overlaps
        out[name] = m
        taken |= m
    out[REGION_ORDER[-1]] = face_mask & ~taken
    return out


def _shares(heat: np.ndarray, masks: dict[str, np.ndarray]) -> list[dict]:
    pos = np.clip(heat, 0, None)
    total = float(pos.sum())
    rows = [
        {"name": n, "share": float(pos[m].sum() / total) if total > 1e-9 else 0.0} for n, m in masks.items()
    ]
    return sorted(rows, key=lambda r: -r["share"])


def _overlay(heat: np.ndarray, vmax: float) -> str:
    """Transparent PNG: red where hiding the area lowers the score, blue where it raises it."""
    heat = cv2.GaussianBlur(heat.astype(np.float32), (0, 0), sigmaX=max(1.0, heat.shape[1] / 40))
    pos = np.clip(heat, 0, None) / vmax
    neg = np.clip(-heat, 0, None) / vmax
    rgba = np.zeros((*heat.shape, 4), np.uint8)  # BGRA
    rgba[..., :3] = np.where(pos[..., None] >= neg[..., None], (30, 70, 255), (255, 140, 40))
    rgba[..., 3] = (np.clip(np.maximum(pos * 0.85, neg * 0.6), 0, 1) * 255).astype(np.uint8)
    return _png_b64(rgba)


def _accumulate(shape, positions, patch_hw, drops) -> np.ndarray:
    heat = np.zeros(shape[:2], np.float32)
    cnt = np.zeros(shape[:2], np.float32)
    ph, pw = patch_hw
    for (y, x), d in zip(positions, drops):
        heat[y : y + ph, x : x + pw] += d
        cnt[y : y + ph, x : x + pw] += 1
    return heat / np.maximum(cnt, 1)


def explain_recognition(engine: FaceEngine, aligned: np.ndarray, refs: np.ndarray) -> dict:
    """Which parts of the aligned face support the match with the stored reference."""
    P, S = 28, 14
    base_emb = engine.embed_aligned(aligned)
    sims0 = refs @ base_emb
    k = int(sims0.argmax())  # the stored photo that matched best (same rule as live matching)
    ref, base = refs[k], float(sims0[k])

    positions = [(y, x) for y in range(0, 112 - P + 1, S) for x in range(0, 112 - P + 1, S)]
    variants = []
    for y, x in positions:
        v = aligned.copy()
        v[y : y + P, x : x + P] = 127  # neutral grey = 0 after ArcFace normalisation
        variants.append(v)
    drops = base - (engine.embed_batch(variants) @ ref)
    heat = _accumulate(aligned.shape, positions, (P, P), drops)

    masks = _region_masks(ALIGNED_KPS, aligned.shape, np.ones(aligned.shape[:2], bool))
    max_drop = float(max(np.abs(heat).max(), 0.0))
    return {
        "base_similarity": base,
        "reference_index": k,
        "max_drop": float(drops.max()),
        "regions": _shares(heat, masks),
        "overlay_png_b64": _overlay(heat, max(max_drop, 0.05)),  # floor so noise isn't blown up
    }


def explain_detection(engine: FaceEngine, crop: np.ndarray, bbox, kps) -> dict:
    """Which parts of the face the detector's confidence depends on (on the stored crop)."""
    h, w = crop.shape[:2]
    x1, y1 = max(int(bbox[0]), 0), max(int(bbox[1]), 0)
    x2, y2 = min(int(bbox[2]), w), min(int(bbox[3]), h)
    target = (float(x1), float(y1), float(x2), float(y2))

    def score_of(img: np.ndarray) -> float:
        best = 0.0
        for d in engine.detect(img):
            if iou(d.box, target) >= 0.3:
                best = max(best, d.score)
        return best

    base = score_of(crop)
    if base <= 0:
        return {"base_score": 0.0, "note": "The detector could not find the face again on this crop"}

    side = min(x2 - x1, y2 - y1)
    P = max(8, int(side / 3.5))
    xs = np.linspace(x1, max(x2 - P, x1), 6).astype(int)
    ys = np.linspace(y1, max(y2 - P, y1), 6).astype(int)
    positions = [(int(y), int(x)) for y in ys for x in xs]
    fill = crop[y1:y2, x1:x2].reshape(-1, 3).mean(axis=0).astype(np.uint8)
    drops = []
    for y, x in positions:
        v = crop.copy()
        v[y : y + P, x : x + P] = fill
        drops.append(base - score_of(v))
    heat = _accumulate(crop.shape, positions, (P, P), np.array(drops))

    face_mask = np.zeros((h, w), bool)
    face_mask[y1:y2, x1:x2] = True
    masks = _region_masks(np.asarray(kps, dtype=np.float32), crop.shape, face_mask)
    max_drop = float(max(drops))
    return {
        "base_score": base,
        "max_drop": max_drop,
        "regions": _shares(heat, masks),
        "overlay_png_b64": _overlay(heat, max(float(np.abs(heat).max()), 0.1)),
        "note": (
            "No single patch changes the score much so the detector is confident" if max_drop < 0.05 else ""
        ),
    }


def explain(engine: FaceEngine, aligned, crop, bbox, kps, refs: np.ndarray) -> dict:
    t0 = time.perf_counter()
    with _gate:
        out = {"method": "occlusion", "model": engine.pack}
        out["recognition"] = explain_recognition(engine, aligned, refs) if len(refs) else None
        out["detection"] = explain_detection(engine, crop, bbox, kps) if crop is not None else None
    out["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
    return out
