"""Environment-backed application configuration."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
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
    embedding_batch_size: int
    top_k: int
    score_threshold: float
    max_upload_mb: int
    request_timeout: float
    jev_mode: str = "off"
    jev_api_key: str = field(default="", repr=False)
    jev_model: str = "jev-1.13.0"
    jev_timeout: float = 5.0
    jev_threshold: float = 0.90

    @classmethod
    def from_env(cls) -> Settings:
        settings = cls(
            chat_provider=os.getenv("RAG_CHAT_PROVIDER", "ollama"),
            embedding_provider=os.getenv("RAG_EMBEDDING_PROVIDER", "ollama"),
            ollama_base_url=os.getenv("RAG_OLLAMA_BASE_URL", "http://localhost:11434"),
            openai_base_url=os.getenv("RAG_OPENAI_BASE_URL", "http://localhost:1234/v1"),
            openai_api_key=os.getenv("RAG_OPENAI_API_KEY", ""),
            chat_model=os.getenv("RAG_CHAT_MODEL", "llama3.2:3b"),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "embeddinggemma"),
            data_dir=Path(os.getenv("RAG_DATA_DIR", "data/qdrant")),
            collection=os.getenv("RAG_COLLECTION", "documents"),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "900")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "150")),
            embedding_batch_size=int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "32")),
            top_k=int(os.getenv("RAG_TOP_K", "4")),
            score_threshold=float(os.getenv("RAG_SCORE_THRESHOLD", "0.15")),
            max_upload_mb=int(os.getenv("RAG_MAX_UPLOAD_MB", "10")),
            request_timeout=float(os.getenv("RAG_REQUEST_TIMEOUT", "120")),
            jev_mode=os.getenv("RAG_JEV_MODE", "off"),
            jev_api_key=os.getenv("TYPESAFE_API_KEY", ""),
            jev_model=os.getenv("RAG_JEV_MODEL", "jev-1.13.0"),
            jev_timeout=float(os.getenv("RAG_JEV_TIMEOUT", "5")),
            jev_threshold=float(os.getenv("RAG_JEV_THRESHOLD", "0.90")),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.jev_mode not in {"off", "observe", "repair"}:
            raise ValueError("Jev mode must be 'off', 'observe', or 'repair'")
        if self.jev_mode != "off" and not self.jev_api_key.strip():
            raise ValueError("TYPESAFE_API_KEY is required for Jev observation or repair")
        if not self.jev_model.strip():
            raise ValueError("Jev model must not be empty")
        if not math.isfinite(self.jev_timeout) or self.jev_timeout <= 0:
            raise ValueError("Jev timeout must be finite and positive")
        if not math.isfinite(self.jev_threshold) or not 0.5 < self.jev_threshold <= 1:
            raise ValueError("Jev threshold must be greater than 0.5 and at most one")
        supported = {"ollama", "openai-compatible"}
        if self.chat_provider not in supported or self.embedding_provider not in supported:
            msg = "providers must be 'ollama' or 'openai-compatible'"
            raise ValueError(msg)
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            msg = "chunk overlap must be non-negative and smaller than chunk size"
            raise ValueError(msg)
        if (
            self.embedding_batch_size <= 0
            or self.top_k <= 0
            or self.max_upload_mb <= 0
            or self.request_timeout <= 0
        ):
            msg = "embedding batch size, top-k, upload size, and request timeout must be positive"
            raise ValueError(msg)
        if not 0 <= self.score_threshold <= 1:
            msg = "score threshold must be between zero and one"
            raise ValueError(msg)
