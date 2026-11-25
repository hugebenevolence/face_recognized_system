import json
import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


_DEFAULT_PRESETS = [
    {"id": "sim", "label": "Simulated camera", "uri": "rtsp://mediamtx:8554/cam1"},
    {"id": "host-webcam", "label": "Host webcam over RTSP", "uri": "rtsp://mediamtx:8554/webcam"},
]


def _presets() -> tuple:
    raw = os.environ.get("SOURCE_PRESETS")
    try:
        return tuple(json.loads(raw)) if raw else tuple(_DEFAULT_PRESETS)
    except ValueError:
        return tuple(_DEFAULT_PRESETS)


@dataclass(frozen=True)
class Config:
    # --- wiring ---
    database_url: str = _env("DATABASE_URL", "postgresql://face:face@db:5432/face")
    backend_url: str = _env("BACKEND_URL", "http://backend:8000")
    internal_token: str = _env("INTERNAL_TOKEN", "change-me")
    data_dir: Path = Path(_env("DATA_DIR", "/data"))
    camera_id: str = _env("CAMERA_ID", "cam1")
    source_uri: str = _env("SOURCE_URI", "rtsp://mediamtx:8554/cam1")
    source_presets: tuple = field(default_factory=_presets)
    # Optional: also publish the annotated video as H.264 over RTSP (e.g. rtsp://mediamtx:8554/annotated)
    rtsp_out: str = _env("RTSP_OUT_URL", "")

    # --- DeepStream graph ---
    config_dir: Path = Path(_env("DS_CONFIG_DIR", "/app/configs"))
    frame_width: int = int(_env("FRAME_WIDTH", "1280"))
    frame_height: int = int(_env("FRAME_HEIGHT", "720"))
    jpeg_quality: int = int(_env("JPEG_QUALITY", "80"))

    # --- models ---
    model_pack: str = _env("MODEL_PACK", "buffalo_l")  # which insightface models are loaded
    # Names the embedding space: the model pack plus how faces are prepared for it. Stored vectors
    # are only compared with vectors of the same id, and the backend re-embeds stored photos when
    # it changes (POST /api/admin/reanalyze). "aligned" = full landmark alignment in nvdspreprocess.
    model_id: str = _env("MODEL_ID", _env("MODEL_PACK", "buffalo_l") + "-aligned")
    det_thresh: float = float(_env("DET_THRESH", "0.5"))
    min_face_px: int = int(_env("MIN_FACE_PX", "40"))

    # --- recognition policy ---
    match_threshold: float = float(_env("MATCH_THRESHOLD", "0.40"))  # cosine similarity
    id_interval: int = int(_env("ID_INTERVAL", "5"))  # frames between ArcFace runs while unresolved
    reid_interval: int = int(_env("REID_INTERVAL", "45"))  # frames between re-checks of known tracks
    unknown_after: int = int(_env("UNKNOWN_AFTER", "4"))  # failed attempts before "Unknown"
    min_hits: int = int(_env("MIN_HITS", "3"))  # frames a track must live before an event
    event_cooldown: float = float(_env("EVENT_COOLDOWN", "30"))  # s before re-logging same track


cfg = Config()
