"""Face detection and embedding for still images, on the DeepStream still-image graph.

The graph (still_graph.py) runs SCRFD, the alignment library and ArcFace on 640x640 views, like
the live graph does on video frames. This module decides which views of a picture to run:
the whole picture (and overlapping tiles of big pictures) to find faces, and one view per face,
centred on it, to embed it. The crop and scale of each view is done on the GPU by the graph.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import Config
from .geometry import TEMPLATE, containment, face_extent, iou
from .picture import VIEW, Picture, View
from .scrfd import Face
from .still_graph import StillFace, StillGraph

TILE_STRIDE = 480  # tiles overlap by 160 px so a face is whole in at least one of them
BIG_IMAGE = 800  # below this the whole-picture view is enough
TILE_LONG_SIDE = 1280  # tiles cut the picture at this size: small faces keep their pixels
MIN_FACE_PX = 24  # in original picture pixels
MAX_ALIGN_PX = 224  # a bigger face is shrunk so its aligned region is this wide in the view
MOSAIC = VIEW // 112  # aligned 112x112 faces per row of a mosaic view


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2, picture pixels
    kps: np.ndarray  # (5, 2) landmarks, picture pixels
    score: float
    aligned: Optional[np.ndarray] = None  # 112x112 RGB the alignment library produced

    @property
    def size(self) -> float:
        return min(self.box[2] - self.box[0], self.box[3] - self.box[1])

    @property
    def area(self) -> float:
        return (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])


def _positions(extent: int) -> list[int]:
    last = max(extent - VIEW, 0)
    pos = list(range(0, last + 1, TILE_STRIDE))
    if pos[-1] != last:
        pos.append(last)
    return pos


class FaceEngine:
    def __init__(self, cfg: Config, graph: StillGraph) -> None:
        self._graph = graph
        self.pack = cfg.model_id  # embedding space id, see Config.model_id

    @property
    def ready(self) -> bool:
        return self._graph.ready

    # ------------------------------------------------------------------ detection
    def _detect_view(self, pic: Picture, view: View) -> list[tuple[Detection, Face]]:
        out = []
        for sf in self._graph.detect(pic, view):
            f = sf.face
            x, y = view.to_picture([f.box[0], f.box[2]], [f.box[1], f.box[3]])
            kx, ky = view.to_picture(f.kps[:, 0], f.kps[:, 1])
            det = Detection((float(x[0]), float(y[0]), float(x[1]), float(y[1])), np.stack([kx, ky], 1), f.score,
                            sf.aligned)
            out.append((det, f))
        return out

    def detect(self, pic: Picture) -> list[Detection]:
        """Faces found with the whole picture fitted into one view."""
        view = View.region(pic, 0, 0, VIEW / max(pic.width, pic.height))
        return [d for d, _ in self._detect_view(pic, view)]

    def detect_region(self, pic: Picture, x0: float, y0: float, x1: float, y1: float, scale: float) -> list[Detection]:
        """Faces in a region of the picture, scaled by `scale` (used to zoom into a selection)."""
        return [d for d, _ in self._detect_view(pic, View.region(pic, x0, y0, scale))]

    def detect_multiscale(self, pic: Picture) -> list[Detection]:
        """The whole-picture view, plus overlapping tiles of a 1280 px version for big pictures
        (small faces in group photos), merged so each face is reported once."""
        found = self.detect(pic)
        if max(pic.width, pic.height) > BIG_IMAGE:
            s = TILE_LONG_SIDE / max(pic.width, pic.height)
            ww, wh = round(pic.width * s), round(pic.height * s)
            for ty in _positions(wh):
                for tx in _positions(ww):
                    view = View.region(pic, tx / s, ty / s, s)
                    tw, th = view.dst[2], view.dst[3]
                    for det, f in self._detect_view(pic, view):
                        if not self._cut_by_tile(f.box, tx, ty, tw, th, ww, wh):
                            found.append(det)
        found = [d for d in found if d.size >= MIN_FACE_PX]
        kept: list[Detection] = []
        for d in sorted(found, key=lambda d: -d.score):
            if all(iou(d.box, k.box) < 0.4 and containment(d.box, k.box) < 0.7 for k in kept):
                kept.append(d)
        return kept

    @staticmethod
    def _cut_by_tile(box, tx, ty, tw, th, ww, wh, margin: float = 3.0) -> bool:
        """A face touching a tile edge that is not the picture edge is only partly in the tile."""
        return (
            (box[0] < margin and tx > 0)
            or (box[1] < margin and ty > 0)
            or (box[2] > tw - margin and tx + tw < ww)
            or (box[3] > th - margin and ty + th < wh)
        )

    # ------------------------------------------------------------------ embedding
    def _face_view(self, pic: Picture, det: Detection) -> tuple[View, Face]:
        """A view centred on the face at its own resolution (a very large face is shrunk)."""
        k = min(1.0, MAX_ALIGN_PX / face_extent(det.kps))
        cx, cy = det.kps[:, 0].mean(), det.kps[:, 1].mean()
        half = VIEW / 2 / k
        view = View.region(pic, cx - half, cy - half, k)
        x, y = view.to_view([det.box[0], det.box[2]], [det.box[1], det.box[3]])
        kx, ky = view.to_view(det.kps[:, 0], det.kps[:, 1])
        face = Face(box=(float(x[0]), float(y[0]), float(x[1]), float(y[1])), kps=np.stack([kx, ky], 1), score=det.score)
        return view, face

    def embed(self, pic: Picture, det: Detection) -> np.ndarray:
        """Embedding of one face, from a view centred on it. SCRFD runs on the view and the
        alignment library aligns the face with its landmarks, exactly as in the live graph.
        The aligned face is kept on the detection (det.aligned)."""
        view, face = self._face_view(pic, det)
        sf = self._graph.embed(pic, view, [face])[0]
        if sf.embedding is None:
            raise RuntimeError("The recogniser returned no embedding")
        if sf.aligned is not None:
            det.aligned = sf.aligned
        return sf.embedding

    def align(self, pic: Picture, det: Detection) -> np.ndarray:
        """The aligned 112x112 face (RGB) the recogniser sees, from the alignment library."""
        if det.aligned is None:
            self.embed(pic, det)
        return det.aligned

    def embed_aligned(self, aligned: np.ndarray) -> np.ndarray:
        return self.embed_batch([aligned])[0]

    def embed_batch(self, aligned_list: list[np.ndarray]) -> np.ndarray:
        """(N, 512) embeddings of already aligned 112x112 RGB faces (explanation maps). They are
        laid out as a mosaic, 25 to a view, as objects of the "aligned face" class, so the
        alignment step only copies each tile and ArcFace embeds every tile in one pass."""
        per_view = MOSAIC * MOSAIC
        full = View(src=(0, 0, VIEW, VIEW), dst=(0, 0, VIEW, VIEW))
        out: list[np.ndarray] = []
        for start in range(0, len(aligned_list), per_view):
            chunk = aligned_list[start : start + per_view]
            canvas, faces = np.zeros((VIEW, VIEW, 3), np.uint8), []
            for i, a in enumerate(chunk):
                y, x = (i // MOSAIC) * 112, (i % MOSAIC) * 112
                canvas[y : y + 112, x : x + 112] = a
                faces.append(Face(box=(x, y, x + 112, y + 112), kps=TEMPLATE + (x, y), score=1.0))
            results: list[StillFace] = self._graph.embed(Picture.from_rgb(canvas), full, faces, prealigned=True)
            if any(r.embedding is None for r in results):
                raise RuntimeError("The recogniser returned no embedding")
            out.extend(r.embedding for r in results)
        return np.stack(out).astype(np.float32)
