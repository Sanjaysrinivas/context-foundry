import pytest

import local_rag.providers as providers
from local_rag.domain import GroundedClaim, GroundedResponse, ProviderError


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
        assert payload["options"] == {"temperature": 0, "seed": 0}
        messages = payload["messages"]
        assert isinstance(messages, list)
        if messages[0]["content"] != "System":
            assert "Do not use background knowledge" in messages[0]["content"]
            assert "reproduce the directly relevant list completely" in messages[0]["content"]
            assert "explicit evidence" in messages[0]["content"]
            assert "citations array" in messages[0]["content"]
        if isinstance(payload.get("format"), dict):
            schema = payload["format"]
            assert isinstance(schema, dict)
            assert "$defs" not in schema
            assert "maxLength" not in repr(schema)
            assert schema["required"] == ["style", "claims", "unsupported"]
            return {
                "message": {
                    "content": '{"style":"paragraphs","claims":'
                    '[{"text":"Grounded","citations":[1]}],"unsupported":[]}'
                }
            }
        if "format" in payload:
            assert payload["format"] == "json"
        return {"message": {"content": "Grounded"}}

    monkeypatch.setattr(providers, "_post", fake_post)
    embeddings = providers.OllamaEmbeddingProvider("http://ollama", "embeddinggemma", 30)
    chat = providers.OllamaChatProvider("http://ollama", "llama3.2:3b", 30)

    assert await embeddings.embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert await chat.answer("Question?", "[1] Evidence", 1) == GroundedResponse(
        claims=[GroundedClaim(text="Grounded", citations=[1])]
    )
    assert await chat.complete("System", "User", json_mode=True) == "Grounded"


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
        response_format = payload.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"style":"bullets","claims":'
                            '[{"text":"Compatible answer","citations":[1]}],'
                            '"unsupported":[]}'
                        }
                    }
                ]
            }
        if "response_format" in payload:
            assert payload["response_format"] == {"type": "json_object"}
        assert payload["temperature"] == 0
        messages = payload["messages"]
        assert isinstance(messages, list)
        if messages[0]["content"] != "System":
            assert "Do not use background knowledge" in messages[0]["content"]
        return {"choices": [{"message": {"content": "Compatible answer"}}]}

    monkeypatch.setattr(providers, "_post", fake_post)
    embeddings = providers.OpenAICompatibleEmbeddingProvider(
        "http://local/v1", "local-key", "embed", 20
    )
    chat = providers.OpenAICompatibleChatProvider("http://local/v1", "local-key", "chat", 20)

    assert await embeddings.embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert await chat.answer("Question?", "Context", 1) == GroundedResponse(
        style="bullets",
        claims=[GroundedClaim(text="Compatible answer", citations=[1])],
    )
    assert await chat.complete("System", "User", json_mode=True) == "Compatible answer"


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
            "Question?", "Context", 1
        )


@pytest.mark.unit
async def test_grounded_answer_retries_invalid_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        '{"style":"paragraphs","claims":[{"text":"Bad","citations":[2]}],"unsupported":[]}',
        '{"style":"paragraphs","claims":[{"text":"Good","citations":[1]}],"unsupported":[]}',
    ]

    async def fake_post(
        url: str,
        payload: dict[str, object],
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        return {"message": {"content": responses.pop(0)}}

    monkeypatch.setattr(providers, "_post", fake_post)
    answer = await providers.OllamaChatProvider("http://local", "chat", 10).answer(
        "Question?", "[1] Evidence", 1
    )

    assert answer.claims[0].text == "Good"
    assert not responses


@pytest.mark.unit
async def test_grounded_answer_enforces_complete_ordered_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        '{"style":"steps","claims":[{"text":"First","citations":[1]}],"unsupported":[]}',
        '{"style":"steps","claims":['
        '{"text":"First","citations":[1]},'
        '{"text":"Second","citations":[1]},'
        '{"text":"Third","citations":[1]}],"unsupported":[]}',
    ]

    async def fake_post(
        url: str,
        payload: dict[str, object],
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        schema = payload["format"]
        assert isinstance(schema, dict)
        properties = schema["properties"]
        assert isinstance(properties, dict)
        claims_schema = properties["claims"]
        assert isinstance(claims_schema, dict)
        assert claims_schema["minItems"] == 3
        return {"message": {"content": responses.pop(0)}}

    monkeypatch.setattr(providers, "_post", fake_post)
    answer = await providers.OllamaChatProvider("http://local", "chat", 10).answer(
        "Trace the complete process", "[1] Report\n1. First\n2. Second\n3. Third", 1
    )

    assert [item.text for item in answer.claims] == ["First", "Second", "Third"]
    assert not responses
