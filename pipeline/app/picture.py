"""Still images and the 640x640 views the still-image graph runs on.

A Picture is either JPEG bytes, which the graph decodes on the GPU (nvjpegdec) and crops and
scales with nvvideoconvert, or an RGB array (other formats, rotated phone photos, images built
in memory such as the explanation mosaics), which is sent as raw pixels.

A View says which rectangle of the picture goes where in the 640x640 frame the detector sees,
and converts coordinates between the two.
"""

import io
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

VIEW = 640  # SCRFD's input size: a view is fed to the network without any rescaling
_EXIF_ORIENTATION = 0x0112


@dataclass
class Picture:
    width: int
    height: int
    jpeg: Optional[bytes] = None  # decoded by the graph on the GPU
    rgb: Optional[np.ndarray] = None  # (H, W, 3) uint8, sent as raw pixels

    @classmethod
    def from_bytes(cls, data: bytes) -> "Picture":
        """An uploaded file. Plain JPEGs stay compressed for the GPU decoder; other formats and
        JPEGs that need an EXIF rotation are decoded here."""
        try:
            img = Image.open(io.BytesIO(data))
            img.size  # noqa: B018  (reads the header only)
        except Exception as exc:
            raise ValueError("Not a valid image") from exc
        orientation = img.getexif().get(_EXIF_ORIENTATION, 1) if img.format == "JPEG" else 1
        if img.format == "JPEG" and orientation == 1 and img.mode in ("RGB", "L", "YCbCr"):
            return cls(width=img.width, height=img.height, jpeg=data)
        img = ImageOps.exif_transpose(img).convert("RGB")
        return cls.from_rgb(np.asarray(img))

    @classmethod
    def from_rgb(cls, rgb: np.ndarray) -> "Picture":
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        return cls(width=rgb.shape[1], height=rgb.shape[0], rgb=rgb)

    def to_rgb(self) -> np.ndarray:
        if self.rgb is not None:
            return self.rgb
        return np.asarray(Image.open(io.BytesIO(self.jpeg)).convert("RGB"))


@dataclass
class View:
    """picture[src] (x, y, w, h, integer pixels) is scaled into view[dst] (x, y, w, h); the rest
    of the 640x640 view is black."""

    src: tuple[int, int, int, int]
    dst: tuple[int, int, int, int]

    @property
    def sx(self) -> float:
        return self.dst[2] / self.src[2]

    @property
    def sy(self) -> float:
        return self.dst[3] / self.src[3]

    def to_view(self, x, y):
        return self.dst[0] + (np.asarray(x) - self.src[0]) * self.sx, self.dst[1] + (np.asarray(y) - self.src[1]) * self.sy

    def to_picture(self, x, y):
        return self.src[0] + (np.asarray(x) - self.dst[0]) / self.sx, self.src[1] + (np.asarray(y) - self.dst[1]) / self.sy

    @classmethod
    def region(cls, pic: Picture, x0: float, y0: float, scale: float, vx: float = 0.0, vy: float = 0.0) -> "View":
        """The picture region whose top-left (x0, y0) lands at (vx, vy) in the view, at `scale`
        view pixels per picture pixel, limited to the picture and to the view."""
        x1, y1 = x0 + (VIEW - vx) / scale, y0 + (VIEW - vy) / scale  # region that fits in the view
        sx0, sy0 = max(0, math.floor(x0)), max(0, math.floor(y0))
        sx1, sy1 = min(pic.width, math.ceil(x1)), min(pic.height, math.ceil(y1))
        if sx1 - sx0 < 2 or sy1 - sy0 < 2:
            raise ValueError("The region is outside the picture")
        dx0 = max(0, min(VIEW - 1, round(vx + (sx0 - x0) * scale)))
        dy0 = max(0, min(VIEW - 1, round(vy + (sy0 - y0) * scale)))
        dw = max(1, min(VIEW - dx0, round((sx1 - sx0) * scale)))
        dh = max(1, min(VIEW - dy0, round((sy1 - sy0) * scale)))
        return cls(src=(sx0, sy0, sx1 - sx0, sy1 - sy0), dst=(dx0, dy0, dw, dh))

    def raw(self, pic: Picture) -> np.ndarray:
        """The 640x640 RGB view built on the CPU, for pictures held as arrays."""
        x, y, w, h = self.src
        part = Image.fromarray(pic.to_rgb()[y : y + h, x : x + w])
        dx, dy, dw, dh = self.dst
        if (dw, dh) != (w, h):
            resample = Image.Resampling.BOX if dw < w else Image.Resampling.BILINEAR
            part = part.resize((dw, dh), resample)
        canvas = np.zeros((VIEW, VIEW, 3), np.uint8)
        canvas[dy : dy + dh, dx : dx + dw] = np.asarray(part)
        return canvas


def png_bytes(rgb: np.ndarray) -> bytes:
    """PNG of an RGB or RGBA array (lossless: aligned faces, overlays)."""
    out = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).save(out, format="PNG")
    return out.getvalue()


def jpeg_bytes(rgb: np.ndarray, quality: int = 90) -> bytes:
    out = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).save(out, format="JPEG", quality=quality)
    return out.getvalue()
