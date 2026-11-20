"""Overlay rendering. cv2.putText can't draw Vietnamese diacritics, so label text is
rendered with Pillow into a small patch and pasted onto the frame."""

import functools

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

GREEN = (80, 200, 60)  # BGR
RED = (60, 60, 230)
AMBER = (0, 190, 255)
_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


@functools.lru_cache(maxsize=4)
def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def draw_label(frame: np.ndarray, box: tuple[float, float, float, float], text: str, color) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    font = _font(max(14, min(22, w // 60)))
    l, t, r, b = font.getbbox(text)
    tw, th = r - l + 10, b - t + 8
    patch = Image.new("RGB", (tw, th), (color[2], color[1], color[0]))  # RGB for PIL
    ImageDraw.Draw(patch).text((5 - l, 4 - t), text, font=font, fill=(255, 255, 255))
    arr = np.asarray(patch)[:, :, ::-1]

    # Place above the box; fall back to inside the box near the top edge.
    px = min(max(x1, 0), max(w - tw, 0))
    py = y1 - th if y1 - th >= 0 else min(y1, h - th)
    pw, ph = min(tw, w - px), min(th, h - py)
    if pw > 0 and ph > 0:
        frame[py : py + ph, px : px + pw] = arr[:ph, :pw]


def draw_landmarks(frame: np.ndarray, kps: np.ndarray) -> None:
    """The 5 points the detector found (eyes, nose, mouth corners): the face is aligned on them."""
    for x, y in kps:
        cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 0), -1)
        cv2.circle(frame, (int(x), int(y)), 3, (0, 230, 255), -1)


def draw_note(frame: np.ndarray, box: tuple[float, float, float, float], text: str) -> None:
    """Small dark caption under the box (explain overlay)."""
    h, w = frame.shape[:2]
    font = _font(max(12, min(16, w // 80)))
    l, t, r, b = font.getbbox(text)
    tw, th = r - l + 8, b - t + 6
    patch = Image.new("RGB", (tw, th), (25, 25, 25))
    ImageDraw.Draw(patch).text((4 - l, 3 - t), text, font=font, fill=(235, 235, 235))
    arr = np.asarray(patch)[:, :, ::-1]
    x1, y2 = int(box[0]), int(box[3])
    px = min(max(x1, 0), max(w - tw, 0))
    py = min(max(y2 + 2, 0), max(h - th, 0))
    pw, ph = min(tw, w - px), min(th, h - py)
    if pw > 0 and ph > 0:
        roi = frame[py : py + ph, px : px + pw]
        frame[py : py + ph, px : px + pw] = (0.78 * arr[:ph, :pw] + 0.22 * roi).astype(np.uint8)


def placeholder(text: str, size: tuple[int, int] = (960, 540)) -> np.ndarray:
    img = Image.new("RGB", size, (24, 26, 32))
    d = ImageDraw.Draw(img)
    font = _font(28)
    l, t, r, b = font.getbbox(text)
    d.text(((size[0] - (r - l)) / 2, (size[1] - (b - t)) / 2), text, font=font, fill=(150, 156, 170))
    return np.asarray(img)[:, :, ::-1].copy()
