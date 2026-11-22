"""Face-quality checks shown in the registration report.

The thresholds are heuristics chosen to flag obviously poor enrolment photos; they are
not calibrated on a dataset and the UI labels pose as an estimate.
"""

import math

import cv2
import numpy as np


def _check(key: str, label: str, value: str, status: str, hint: str) -> dict:
    return {"key": key, "label": label, "value": value, "status": status, "hint": hint}


def assess(aligned: np.ndarray, box, kps, det_score: float) -> dict:
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())  # variance of Laplacian: low = blurry
    brightness = float(gray.mean())
    size = float(min(box[2] - box[0], box[3] - box[1]))

    le, re_, nose = kps[0], kps[1], kps[2]
    iod = max(math.dist(le, re_), 1e-6)  # inter-ocular distance
    yaw = float((nose[0] - (le[0] + re_[0]) / 2) / iod)  # ~0 when the face looks straight ahead
    roll = math.degrees(math.atan2(re_[1] - le[1], re_[0] - le[0]))

    checks = [
        _check(
            "size",
            "Face size",
            f"{size:.0f} px",
            "fail" if size < 60 else "warn" if size < 100 else "ok",
            "Below the 60 px minimum"
            if size < 60
            else "Needs at least 100 px"
            if size < 100
            else "Large enough",
        ),
        _check(
            "det",
            "Detection confidence",
            f"{det_score:.2f}",
            "warn" if det_score < 0.7 else "ok",
            "Low confidence" if det_score < 0.7 else "Confident",
        ),
        _check(
            "blur",
            "Sharpness",
            f"{blur:.0f}",
            "warn" if blur < 40 else "ok",
            "Blurry" if blur < 40 else "Sharp enough",
        ),
        _check(
            "light",
            "Brightness",
            f"{brightness:.0f}",
            "warn" if brightness < 60 or brightness > 200 else "ok",
            "Too dark" if brightness < 60 else "Too bright" if brightness > 200 else "Lighting is fine",
        ),
        _check(
            "pose",
            "Head turn estimate",
            f"{abs(yaw) * 100:.0f}%",
            "warn" if abs(yaw) > 0.25 else "ok",
            "Face is turned" if abs(yaw) > 0.25 else "Facing the camera",
        ),
    ]
    statuses = {c["status"] for c in checks}
    overall = "fail" if "fail" in statuses else "warn" if "warn" in statuses else "ok"
    return {
        "overall": overall,
        "checks": checks,
        "metrics": {
            "size_px": size,
            "det_score": float(det_score),
            "blur": blur,
            "brightness": brightness,
            "yaw_ratio": yaw,
            "roll_deg": float(roll),
        },
    }
