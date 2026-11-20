from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://face:face@db:5432/face"
    pipeline_url: str = "http://pipeline:8090"
    internal_token: str = "change-me"
    data_dir: Path = Path("/data")
    cors_origins: str = "http://localhost:3000"
    camera_id: str = "cam1"
    max_upload_mb: int = 10
    # Registering a face at least this similar to a *different* person needs confirmation.
    dup_threshold: float = 0.5
    # Adding a photo to a person whose similarity to all their existing photos is below this
    # needs confirmation (probably a different person).
    consistency_threshold: float = 0.3

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
