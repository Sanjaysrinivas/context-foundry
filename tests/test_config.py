from dataclasses import replace
from pathlib import Path

import pytest

from local_rag.config import Settings
from local_rag.factory import build_chat_provider, build_embedding_provider, build_service
from local_rag.providers import (
    OllamaChatProvider,
    OllamaEmbeddingProvider,
    OpenAICompatibleChatProvider,
    OpenAICompatibleEmbeddingProvider,
)


@pytest.mark.unit
def test_local_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "RAG_CHAT_PROVIDER",
        "RAG_EMBEDDING_PROVIDER",
        "RAG_CHAT_MODEL",
        "RAG_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.chat_provider == "ollama"
    assert settings.embedding_provider == "ollama"
    assert settings.chat_model == "llama3.2:3b"
    assert settings.embedding_model == "embeddinggemma"
    assert settings.embedding_batch_size == 32
    assert settings.score_threshold == 0.15


@pytest.mark.unit
@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (replace(Settings.from_env(), chat_provider="unknown"), "providers"),
        (replace(Settings.from_env(), chunk_overlap=900), "chunk overlap"),
        (replace(Settings.from_env(), top_k=0), "must be positive"),
        (replace(Settings.from_env(), embedding_batch_size=0), "must be positive"),
        (
            replace(Settings.from_env(), score_threshold=1.1),
            "between zero and one",
        ),
    ],
)
def test_rejects_invalid_settings(settings: Settings, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        settings.validate()


@pytest.mark.unit
def test_builds_each_provider_independently() -> None:
    local = Settings.from_env()
    compatible = replace(
        local,
        chat_provider="openai-compatible",
        embedding_provider="openai-compatible",
    )

    assert isinstance(build_chat_provider(local), OllamaChatProvider)
    assert isinstance(build_embedding_provider(local), OllamaEmbeddingProvider)
    assert isinstance(build_chat_provider(compatible), OpenAICompatibleChatProvider)
    assert isinstance(build_embedding_provider(compatible), OpenAICompatibleEmbeddingProvider)


@pytest.mark.integration
def test_builds_complete_local_service(tmp_path: Path) -> None:
    settings = replace(Settings.from_env(), data_dir=tmp_path / "qdrant")

    service = build_service(settings)

    assert isinstance(service.embedding_provider, OllamaEmbeddingProvider)
    assert isinstance(service.chat_provider, OllamaChatProvider)
    service.close()
