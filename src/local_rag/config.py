"""Environment-backed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    chat_provider: str
    embedding_provider: str
    ollama_base_url: str
    openai_base_url: str
    openai_api_key: str
    chat_model: str
    embedding_model: str
    data_dir: Path
    collection: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    score_threshold: float
    max_upload_mb: int
    request_timeout: float

    @classmethod
    def from_env(cls) -> Settings:
        settings = cls(
            chat_provider=os.getenv("RAG_CHAT_PROVIDER", "ollama"),
            embedding_provider=os.getenv("RAG_EMBEDDING_PROVIDER", "ollama"),
            ollama_base_url=os.getenv("RAG_OLLAMA_BASE_URL", "http://localhost:11434"),
            openai_base_url=os.getenv(
                "RAG_OPENAI_BASE_URL", "http://localhost:1234/v1"
            ),
            openai_api_key=os.getenv("RAG_OPENAI_API_KEY", ""),
            chat_model=os.getenv("RAG_CHAT_MODEL", "llama3.2:3b"),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "embeddinggemma"),
            data_dir=Path(os.getenv("RAG_DATA_DIR", "data/qdrant")),
            collection=os.getenv("RAG_COLLECTION", "documents"),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "900")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "150")),
            top_k=int(os.getenv("RAG_TOP_K", "4")),
            score_threshold=float(os.getenv("RAG_SCORE_THRESHOLD", "0.25")),
            max_upload_mb=int(os.getenv("RAG_MAX_UPLOAD_MB", "10")),
            request_timeout=float(os.getenv("RAG_REQUEST_TIMEOUT", "120")),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        supported = {"ollama", "openai-compatible"}
        if (
            self.chat_provider not in supported
            or self.embedding_provider not in supported
        ):
            msg = "providers must be 'ollama' or 'openai-compatible'"
            raise ValueError(msg)
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            msg = "chunk overlap must be non-negative and smaller than chunk size"
            raise ValueError(msg)
        if self.top_k <= 0 or self.max_upload_mb <= 0 or self.request_timeout <= 0:
            msg = "top-k, upload size, and request timeout must be positive"
            raise ValueError(msg)
        if not 0 <= self.score_threshold <= 1:
            msg = "score threshold must be between zero and one"
            raise ValueError(msg)
