"""Decodes the raw SCRFD output tensors (as attached by nvinfer) into faces.

SCRFD predicts, for every anchor on three feature maps (strides 8/16/32), a face score,
four distances to the box edges and ten landmark offsets. nvinfer already ran the network
on TensorRT; this only turns its output into boxes + five landmarks.
"""

from dataclasses import dataclass

import numpy as np

NET_SIZE = 640
STRIDES = (8, 16, 32)
ANCHORS_PER_CELL = 2
NMS_IOU = 0.4


@dataclass
class Face:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in frame pixels
    kps: np.ndarray  # (5, 2) landmarks: eyes, nose, mouth corners
    score: float

    @property
    def size(self) -> float:
        return min(self.box[2] - self.box[0], self.box[3] - self.box[1])


def _centers(stride: int) -> np.ndarray:
    n = NET_SIZE // stride
    ys, xs = np.mgrid[:n, :n]
    grid = np.stack([xs, ys], axis=-1).reshape(-1, 2) * stride
    return np.repeat(grid, ANCHORS_PER_CELL, axis=0).astype(np.float32)


_CENTERS = {s: _centers(s) for s in STRIDES}


def _nms(boxes: np.ndarray, scores: np.ndarray) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= NMS_IOU]
    return keep


def decode(tensors: dict[tuple[int, int], np.ndarray], frame_w: int, frame_h: int, thresh: float) -> list[Face]:
    """`tensors` maps (anchor_count, channels) -> array; channels are 1 / 4 / 10 for
    score / box / landmarks. nvinfer letterboxes the frame into the 640x640 input at the
    top-left corner (symmetric-padding=0), so coordinates divide by a single scale."""
    scale = min(NET_SIZE / frame_w, NET_SIZE / frame_h)
    boxes, kpss, scores = [], [], []
    for stride in STRIDES:
        n = (NET_SIZE // stride) ** 2 * ANCHORS_PER_CELL
        sc, bb, kp = tensors.get((n, 1)), tensors.get((n, 4)), tensors.get((n, 10))
        if sc is None or bb is None or kp is None:
            continue
        sc = sc.reshape(-1)
        idx = np.where(sc >= thresh)[0]
        if idx.size == 0:
            continue
        c = _CENTERS[stride][idx]
        d = bb[idx] * stride
        boxes.append(np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], axis=1))
        k = kp[idx].reshape(-1, 5, 2) * stride + c[:, None, :]
        kpss.append(k)
        scores.append(sc[idx])
    if not boxes:
        return []
    boxes, kpss, scores = np.concatenate(boxes), np.concatenate(kpss), np.concatenate(scores)
    keep = _nms(boxes, scores)
    faces = []
    for i in keep:
        b = boxes[i] / scale
        faces.append(Face(box=(float(b[0]), float(b[1]), float(b[2]), float(b[3])), kps=kpss[i] / scale, score=float(scores[i])))
    return faces
