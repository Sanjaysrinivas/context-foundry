import pytest

import local_rag.providers as providers
from local_rag.domain import ProviderError


@pytest.mark.unit
async def test_ollama_providers_parse_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_post(
        url: str,
        payload: dict[str, object],
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        assert timeout == 30
        assert headers is None
        if url.endswith("/api/embed"):
            assert payload["model"] == "embeddinggemma"
            return {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
        assert payload["model"] == "llama3.2:3b"
        return {"message": {"content": "Grounded [1]"}}

    monkeypatch.setattr(providers, "_post", fake_post)
    embeddings = providers.OllamaEmbeddingProvider("http://ollama", "embeddinggemma", 30)
    chat = providers.OllamaChatProvider("http://ollama", "llama3.2:3b", 30)

    assert await embeddings.embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert await chat.answer("Question?", "[1] Evidence") == "Grounded [1]"


@pytest.mark.unit
async def test_compatible_providers_parse_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_post(
        url: str,
        payload: dict[str, object],
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        assert headers == {"Authorization": "Bearer local-key"}
        if url.endswith("/embeddings"):
            return {
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            }
        return {"choices": [{"message": {"content": "Compatible answer"}}]}

    monkeypatch.setattr(providers, "_post", fake_post)
    embeddings = providers.OpenAICompatibleEmbeddingProvider(
        "http://local/v1", "local-key", "embed", 20
    )
    chat = providers.OpenAICompatibleChatProvider("http://local/v1", "local-key", "chat", 20)

    assert await embeddings.embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert await chat.answer("Question?", "Context") == "Compatible answer"


@pytest.mark.unit
async def test_rejects_malformed_provider_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_post(
        url: str,
        payload: dict[str, object],
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        return {}

    monkeypatch.setattr(providers, "_post", fake_post)

    with pytest.raises(ProviderError, match="invalid embedding"):
        await providers.OllamaEmbeddingProvider("http://local", "embed", 10).embed(["a"])
    with pytest.raises(ProviderError, match="invalid chat"):
        await providers.OpenAICompatibleChatProvider("http://local/v1", "", "chat", 10).answer(
            "Question?", "Context"
        )
