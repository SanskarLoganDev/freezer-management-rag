"""Validated settings for the planned local RAG pipeline.

Run from the project root so relative data paths and .env resolve consistently.
Loading settings does not contact Ollama or create an index.
"""

from pathlib import Path

from pydantic import AnyHttpUrl, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment variables use the RAG_ prefix; process values override .env."""

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    ollama_base_url: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:11434")
    llm_model: str = Field(default="qwen3.5:4b", min_length=1)
    embedding_model: str = Field(default="nomic-embed-text:v1.5", min_length=1)
    embedding_dimensions: int = Field(default=768, gt=0)
    embedding_batch_size: int = Field(default=4, ge=1)
    embedding_keep_alive: str = "0"
    llm_keep_alive: str = "1m"
    llm_num_ctx: int = Field(default=4096, ge=1024)
    llm_num_predict: int = Field(default=512, ge=1)
    llm_temperature: float = Field(default=0.1, ge=0, le=2)

    documents_dir: Path = Path("docs")
    processed_dir: Path = Path("data/processed")
    manifests_dir: Path = Path("data/manifests")
    qdrant_path: Path = Path("data/indexes/qdrant")
    qdrant_collection: str = Field(default="freezer_docs_nomic_v1_5_768", min_length=1)

    chunk_size_chars: int = Field(default=1600, ge=1)
    chunk_overlap_chars: int = Field(default=200, ge=0)
    dense_top_k: int = Field(default=8, ge=1)
    lexical_top_k: int = Field(default=8, ge=1)
    final_top_k: int = Field(default=4, ge=1)
    max_context_chars: int = Field(default=8000, ge=1)
    request_timeout_seconds: float = Field(default=120, gt=0)

    @model_validator(mode="after")
    def validate_retrieval_settings(self) -> "Settings":
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError("chunk_overlap_chars must be less than chunk_size_chars")
        if self.final_top_k > self.dense_top_k + self.lexical_top_k:
            raise ValueError("final_top_k exceeds the total retrieval candidate budget")
        return self
