"""SCRFD detection + ArcFace embedding via insightface's ONNX models."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

from .config import Config

log = logging.getLogger("engine")


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    kps: np.ndarray  # (5, 2) landmarks, used for alignment
    score: float

    @property
    def size(self) -> float:
        return min(self.box[2] - self.box[0], self.box[3] - self.box[1])

    @property
    def area(self) -> float:
        return (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])


class FaceEngine:
    def __init__(self, cfg: Config) -> None:
        self._pack = cfg.model_pack
        use_gpu = "CUDAExecutionProvider" in cfg.providers
        self._app = FaceAnalysis(
            name=cfg.model_pack,
            allowed_modules=["detection", "recognition"],
            providers=list(cfg.providers),
        )
        self._app.prepare(
            ctx_id=0 if use_gpu else -1,
            det_thresh=cfg.det_thresh,
            det_size=(cfg.det_size, cfg.det_size),
        )
        self._det = self._app.models["detection"]
        self._rec = self._app.models["recognition"]
        log.info("FaceEngine ready (pack=%s providers=%s)", cfg.model_pack, cfg.providers)

    @property
    def pack(self) -> str:
        return self._pack

    def detect(self, img_bgr: np.ndarray, size: Optional[int] = None) -> list[Detection]:
        """`size` overrides the detector input size for this call (stills / group photos
        need a larger one than live video or small faces vanish when downscaled)."""
        bboxes, kpss = self._det.detect(
            img_bgr,
            input_size=(size, size) if size else None,
            max_num=0,
            metric="default",
        )
        if bboxes is None or len(bboxes) == 0:
            return []
        return [
            Detection(
                box=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                kps=kpss[i],
                score=float(b[4]),
            )
            for i, b in enumerate(bboxes)
        ]

    @staticmethod
    def align(img_bgr: np.ndarray, det: Detection) -> np.ndarray:
        """112x112 face aligned on the 5 landmarks: exactly what the recogniser sees."""
        return face_align.norm_crop(img_bgr, landmark=det.kps, image_size=112)

    def embed_aligned(self, aligned: np.ndarray) -> np.ndarray:
        """L2-normalised 512-d ArcFace embedding of an already aligned 112x112 face."""
        return self.embed_batch([aligned])[0]

    def embed_batch(self, aligned_list: list[np.ndarray]) -> np.ndarray:
        """(N, 512) L2-normalised embeddings; one ONNX call, used by the heatmaps."""
        feats = np.asarray(self._rec.get_feat(list(aligned_list)), dtype=np.float32)
        feats = feats.reshape(len(aligned_list), -1)
        return feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)

    def embed(self, img_bgr: np.ndarray, det: Detection) -> np.ndarray:
        return self.embed_aligned(self.align(img_bgr, det))
