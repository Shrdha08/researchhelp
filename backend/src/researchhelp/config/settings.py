"""Central configuration. Every tunable (models, K values, chunk sizes) lives here so that
experiments change configuration, not code."""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
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
    llm_model: str = "openai/gpt-oss-120b"
    router_model: str = "openai/gpt-oss-20b"
    llm_temperature: float = 0.0
    # gpt-oss models reason before answering; "low" is enough for grounded extraction and
    # saves tokens (they count against Groq's per-minute budget) and latency.
    llm_reasoning_effort: str | None = "low"
    # Evaluation judge: a different model family than the generator to reduce self-preference.
    judge_model: str = "qwen/qwen3.8-27b"
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
    # Chosen on the dev split (docs/evaluation.md): best Recall@5 and fastest of the 3 tried.
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # Where the ONNX models run: auto (CUDA if available, else CPU) | cuda | cpu.
    onnx_device: str = "auto"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Retrieval hyperparameters (evaluated in Phase 2)
    retrieval_strategy: str = "hybrid_rerank"  # semantic | hybrid | hybrid_rerank
    min_per_paper: int = 1
    k_per_paper: int = 8
    k_candidates: int = 10
    k_final: int = 6

    # Research assistant (Phase 3): a second, fixed retrieval query surfaces the passages where
    # papers discuss their own weaknesses; the merged context is capped at research_k_final.
    research_k_final: int = 10
    research_aux_query: str = (
        "limitations, weaknesses, assumptions, failure cases, future work, open problems"
    )

    # API (Phase 4)
    max_upload_mb: int = 50
    cors_origins: list[str] = ["http://localhost:5173"]  # Vite dev server (Phase 6)

    # Paths
    data_dir: Path = REPO_ROOT / "data"

    @field_validator("qdrant_path", "data_dir")
    @classmethod
    def _relative_to_repo_root(cls, value: Path | None) -> Path | None:
        """Relative paths in .env mean "relative to the repo root", not to the current
        directory, so the CLI behaves the same from backend/ or the root."""
        if value is None or value.is_absolute():
            return value
        return (REPO_ROOT / value).resolve()

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def registry_path(self) -> Path:
        """JSON paper registry used until PostgreSQL replaces it in Phase 5."""
        return self.data_dir / "papers.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
