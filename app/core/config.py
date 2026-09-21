from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "DeepGuard-NR API"
    debug: bool = True
    secret_key: str = "change_me_for_production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 120
    database_url: str = "sqlite:///./deepguard_nr.db"
    cors_origins: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:9090",
        "http://127.0.0.1:9090",
        "https://deepguard-nr.netlify.app/"
        "arise-apostle-salsa.ngrok-free.dev"
    ]
    max_frames: int = 1024
    upload_dir: str = "storage/uploads"
    heatmap_dir: str = "storage/heatmaps"
    checkpoint_path: str = "models/deepguard_nr_checkpoint.pt"
    supported_formats: List[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, value):
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

    max_video_size_mb: int = 500
    max_video_duration_minutes: int = 10

    midas_model_type: str = "MiDaS_small"

    use_mixed_precision: bool = True

    mamba_hidden_dim: int = 128

    enable_parallel_modules: bool = True


settings = Settings()
