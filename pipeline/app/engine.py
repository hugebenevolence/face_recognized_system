"""Face detection and embedding for still images, on the DeepStream still-image graph.

The graph (still_graph.py) takes 640x640 views and runs SCRFD, the alignment library and ArcFace
on them, like the live graph does on video frames. This module turns an arbitrary photo into
views: the whole photo (and overlapping tiles of big photos) to find faces, and one view per face
to embed it.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .geometry import TEMPLATE, aligned_crop, containment, face_extent, iou
from .scrfd import Face
from .still_graph import VIEW, StillGraph

TILE_STRIDE = 480  # tiles overlap by 160 px so a face is whole in at least one of them
BIG_IMAGE = 800  # below this the whole-image view is enough
TILE_LONG_SIDE = 1280  # tiles cut the image at this size: small faces keep their pixels
MIN_FACE_PX = 24  # in original image pixels
MAX_ALIGN_PX = 224  # a bigger face is shrunk (area averaging) so its aligned region is this wide
MOSAIC = VIEW // 112  # aligned 112x112 faces per row of a mosaic view


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


def _canvas() -> np.ndarray:
    return np.zeros((VIEW, VIEW, 3), np.uint8)


def _resize(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    size = (max(int(round(w * scale)), 1), max(int(round(h * scale)), 1))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def _positions(extent: int) -> list[int]:
    last = max(extent - VIEW, 0)
    pos = list(range(0, last + 1, TILE_STRIDE))
    if pos[-1] != last:
        pos.append(last)
    return pos


def _map_face(f: Face, scale: float, dx: float = 0.0, dy: float = 0.0) -> Detection:
    """A face found in a view, back in original image pixels."""
    box = (f.box[0] + dx, f.box[1] + dy, f.box[2] + dx, f.box[3] + dy)
    return Detection(tuple(float(v / scale) for v in box), (f.kps + (dx, dy)) / scale, f.score)


class FaceEngine:
    def __init__(self, cfg: Config, graph: StillGraph) -> None:
        self._graph = graph
        self.pack = cfg.model_id  # embedding space id, see Config.model_id

    @property
    def ready(self) -> bool:
        return self._graph.ready

    # ------------------------------------------------------------------ detection
    def detect(self, img_bgr: np.ndarray, size=None) -> list[Detection]:
        """Faces found with the whole image fitted into one view. `size` is accepted for
        compatibility and ignored: the view size is the detector's input size."""
        scale = VIEW / max(img_bgr.shape[:2])
        view = _canvas()
        small = _resize(img_bgr, scale)
        view[: small.shape[0], : small.shape[1]] = small
        return [_map_face(s.face, scale) for s in self._graph.detect(view)]

    def detect_multiscale(self, img_bgr: np.ndarray) -> list[Detection]:
        """The whole-image view, plus overlapping tiles of a 1280 px version for big images
        (small faces in group photos), merged so each face is reported once."""
        found = self.detect(img_bgr)
        h, w = img_bgr.shape[:2]
        if max(h, w) > BIG_IMAGE:
            s = TILE_LONG_SIDE / max(h, w)
            work = _resize(img_bgr, s)
            wh, ww = work.shape[:2]
            for ty in _positions(wh):
                for tx in _positions(ww):
                    tile = work[ty : ty + VIEW, tx : tx + VIEW]
                    view = _canvas()
                    view[: tile.shape[0], : tile.shape[1]] = tile
                    for st in self._graph.detect(view):
                        if not self._cut_by_tile(st.face.box, tx, ty, tile.shape[1], tile.shape[0], ww, wh):
                            found.append(_map_face(st.face, s, tx, ty))
        found = [d for d in found if d.size >= MIN_FACE_PX]
        kept: list[Detection] = []
        for d in sorted(found, key=lambda d: -d.score):
            if all(iou(d.box, k.box) < 0.4 and containment(d.box, k.box) < 0.7 for k in kept):
                kept.append(d)
        return kept

    @staticmethod
    def _cut_by_tile(box, tx, ty, tw, th, ww, wh, margin: float = 3.0) -> bool:
        """A face touching a tile edge that is not the image edge is only partly in the tile."""
        return (
            (box[0] < margin and tx > 0)
            or (box[1] < margin and ty > 0)
            or (box[2] > tw - margin and tx + tw < ww)
            or (box[3] > th - margin and ty + th < wh)
        )

    # ------------------------------------------------------------------ embedding
    @staticmethod
    def align(img_bgr: np.ndarray, det: Detection) -> np.ndarray:
        """112x112 face aligned on the 5 landmarks: what the recogniser sees."""
        return aligned_crop(img_bgr, det.kps)

    def embed(self, img_bgr: np.ndarray, det: Detection) -> np.ndarray:
        """Embedding of one face. The face is centred in a view at its own resolution (a very
        large face is shrunk first), and the graph aligns and embeds it with its landmarks,
        exactly as the live graph does on a video frame."""
        k = min(1.0, MAX_ALIGN_PX / face_extent(det.kps))
        img = _resize(img_bgr, k) if k < 1.0 else img_bgr
        kps = det.kps * k
        dx, dy = VIEW / 2 - kps[:, 0].mean(), VIEW / 2 - kps[:, 1].mean()
        shift = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        view = cv2.warpAffine(img, shift, (VIEW, VIEW), flags=cv2.INTER_NEAREST, borderValue=0)
        box = tuple(float(v * k + o) for v, o in zip(det.box, (dx, dy, dx, dy)))
        emb = self._graph.embed(view, [Face(box=box, kps=kps + (dx, dy), score=det.score)])[0]
        if emb is None:
            raise RuntimeError("The recogniser returned no embedding")
        return emb

    def embed_aligned(self, aligned: np.ndarray) -> np.ndarray:
        return self.embed_batch([aligned])[0]

    def embed_batch(self, aligned_list: list[np.ndarray]) -> np.ndarray:
        """(N, 512) embeddings of already aligned 112x112 faces (used by the explanation maps).
        They are laid out as a mosaic, 25 to a view, as objects of the "aligned face" class, so
        the alignment step only copies each tile and ArcFace embeds every tile in one pass."""
        per_view = MOSAIC * MOSAIC
        out: list[np.ndarray] = []
        for start in range(0, len(aligned_list), per_view):
            chunk = aligned_list[start : start + per_view]
            view, faces = _canvas(), []
            for i, a in enumerate(chunk):
                y, x = (i // MOSAIC) * 112, (i % MOSAIC) * 112
                view[y : y + 112, x : x + 112] = a
                faces.append(Face(box=(x, y, x + 112, y + 112), kps=TEMPLATE + (x, y), score=1.0))
            embs = self._graph.embed(view, faces, prealigned=True)
            if any(e is None for e in embs):
                raise RuntimeError("The recogniser returned no embedding")
            out.extend(embs)
        return np.stack(out).astype(np.float32)
