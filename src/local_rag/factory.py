"""Construct configured provider and storage implementations."""

from __future__ import annotations

from local_rag.config import Settings
from local_rag.providers import (
    ChatProvider,
    EmbeddingProvider,
    OllamaChatProvider,
    OllamaEmbeddingProvider,
    OpenAICompatibleChatProvider,
    OpenAICompatibleEmbeddingProvider,
)
from local_rag.service import RAGService
from local_rag.store import QdrantVectorStore


def build_service(settings: Settings) -> RAGService:
    return RAGService(
        build_embedding_provider(settings),
        build_chat_provider(settings),
        QdrantVectorStore(settings.data_dir, settings.collection),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        embedding_batch_size=settings.embedding_batch_size,
        top_k=settings.top_k,
        score_threshold=settings.score_threshold,
    )


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "ollama":
        return OllamaEmbeddingProvider(
            settings.ollama_base_url,
            settings.embedding_model,
            settings.request_timeout,
        )
    return OpenAICompatibleEmbeddingProvider(
        settings.openai_base_url,
        settings.openai_api_key,
        settings.embedding_model,
        settings.request_timeout,
    )


def build_chat_provider(settings: Settings) -> ChatProvider:
    if settings.chat_provider == "ollama":
        return OllamaChatProvider(
            settings.ollama_base_url,
            settings.chat_model,
            settings.request_timeout,
        )
    return OpenAICompatibleChatProvider(
        settings.openai_base_url,
        settings.openai_api_key,
        settings.chat_model,
        settings.request_timeout,
    )
