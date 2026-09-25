"""Central configuration. Every tunable (models, K values, chunk sizes) lives here so that
experiments change configuration, not code."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = backend/src/researchhelp/config/settings.py -> parents[4]
REPO_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM (Groq)
    groq_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"
    router_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.0
    llm_max_retries: int = 5

    # Stores
    qdrant_url: str = "http://localhost:6333"
    # If set, use embedded on-disk Qdrant instead of the server (handy without Docker).
    qdrant_path: Path | None = None
    qdrant_collection: str = "paper_chunks"
    database_url: str = "postgresql+psycopg://researchhelp:researchhelp@localhost:5432/researchhelp"

    # Local models (FastEmbed / ONNX)
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "BAAI/bge-reranker-base"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Retrieval hyperparameters (evaluated in Phase 2)
    k_per_paper: int = 8
    k_candidates: int = 20
    k_final: int = 6

    # Paths
    data_dir: Path = REPO_ROOT / "data"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"


@lru_cache
def get_settings() -> Settings:
    return Settings()
