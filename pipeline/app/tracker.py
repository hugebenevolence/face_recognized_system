"""Minimal IoU tracker: gives each face a stable id across frames so we only run the
(expensive) recognition step occasionally and log one event per appearance."""

from dataclasses import dataclass, field
from typing import Optional

Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    box: Box
    hits: int = 1
    misses: int = 0
    # identity: pending -> known | unknown
    status: str = "pending"
    person_id: Optional[int] = None
    name: Optional[str] = None
    code: Optional[str] = None
    score: float = 0.0
    attempts: int = 0
    last_attempt_frame: int = -(10**9)
    last_event_ts: float = 0.0
    last_event_status: Optional[str] = None
    extra: dict = field(default_factory=dict)


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 20) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._tracks: list[Track] = []
        self._next_id = 1

    def update(self, boxes: list[Box]) -> list[Track]:
        """Associate detections to tracks. Returns one Track per input box, same order."""
        pairs = sorted(
            ((iou(t.box, b), ti, bi) for ti, t in enumerate(self._tracks) for bi, b in enumerate(boxes)),
            reverse=True,
        )
        used_t: set[int] = set()
        assigned: dict[int, Track] = {}
        for score, ti, bi in pairs:
            if score < self.iou_threshold:
                break
            if ti in used_t or bi in assigned:
                continue
            used_t.add(ti)
            t = self._tracks[ti]
            t.box, t.hits, t.misses = boxes[bi], t.hits + 1, 0
            assigned[bi] = t

        for ti, t in enumerate(self._tracks):
            if ti not in used_t:
                t.misses += 1

        for bi, b in enumerate(boxes):
            if bi not in assigned:
                t = Track(id=self._next_id, box=b)
                self._next_id += 1
                self._tracks.append(t)
                assigned[bi] = t

        self._tracks = [t for t in self._tracks if t.misses <= self.max_age]
        return [assigned[i] for i in range(len(boxes))]
