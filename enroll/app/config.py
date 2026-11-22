import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Config:
    database_url: str = _env("DATABASE_URL", "postgresql://face:face@db:5432/face")

    # insightface model pack and ONNX Runtime providers (CPU by default, CUDA when available)
    model_pack: str = _env("MODEL_PACK", "buffalo_l")
    providers: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            p.strip() for p in _env("ORT_PROVIDERS", "CPUExecutionProvider").split(",") if p.strip()
        )
    )
    det_size: int = int(_env("DET_SIZE", "640"))
    det_thresh: float = float(_env("DET_THRESH", "0.5"))

    match_threshold: float = float(_env("MATCH_THRESHOLD", "0.40"))  # cosine similarity


cfg = Config()
