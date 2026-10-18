from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma:2b"
    ollama_timeout_seconds: float = 180.0
    ollama_health_timeout_seconds: float = 3.0

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    data_dir: Path = Path("data")
    upload_dir: Path = Path("uploads")
    chroma_dir: Optional[Path] = None
    sqlite_path: Optional[Path] = None

    max_upload_mb: int = 25
    chunk_size: int = 1000
    chunk_overlap: int = 150

    top_k: int = 4
    relevance_threshold: float = 0.25

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @field_validator("data_dir", "upload_dir", "chroma_dir", "sqlite_path")
    @classmethod
    def _resolve_relative(cls, value):
        if value is None:
            return value
        path = Path(value)
        return path if path.is_absolute() else (BACKEND_DIR / path).resolve()

    @property
    def chroma_path(self) -> Path:
        return self.chroma_dir or (self.data_dir / "chroma")

    @property
    def sqlite_file(self) -> Path:
        return self.sqlite_path or (self.data_dir / "app.db")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
