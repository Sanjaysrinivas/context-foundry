from dataclasses import replace

from fastapi.testclient import TestClient

from local_rag.api import create_app
from local_rag.config import Settings
from local_rag.domain import Chunk, SearchResult
from local_rag.service import RAGService


class FakeEmbeddings:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(len(text))] for text in texts]


class FakeChat:
    async def answer(self, question: str, context: str) -> str:
        assert question and "notes.txt" in context
        return "The evidence stays local [1]."


class FakeStore:
    def __init__(self) -> None:
        self.closed = False
        self.cleared = False

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        assert chunks and vectors

    def search(self, vector: list[float], limit: int, threshold: float) -> list[SearchResult]:
        return [SearchResult("notes.txt", 1, "Evidence stays local.", 0.92)]

    def clear(self) -> None:
        self.cleared = True

    def close(self) -> None:
        self.closed = True


def test_web_api_flow() -> None:
    settings = replace(Settings.from_env(), max_upload_mb=1)
    store = FakeStore()
    service = RAGService(
        FakeEmbeddings(),
        FakeChat(),
        store,
        chunk_size=50,
        chunk_overlap=5,
        top_k=4,
        score_threshold=0.25,
    )

    with TestClient(create_app(settings, service)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").json()["chat_provider"] == "ollama"

        upload = client.post(
            "/api/documents",
            files={"file": ("notes.txt", b"Evidence stays local.", "text/plain")},
        )
        assert upload.status_code == 200
        assert upload.json() == {"filename": "notes.txt", "chunks": 1}

        query = client.post("/api/query", json={"question": "Where is the evidence?"})
        assert query.status_code == 200
        assert query.json()["citations"][0]["score"] == 0.92

        assert client.delete("/api/documents").status_code == 200
        assert store.cleared

    assert store.closed


def test_upload_validation_returns_client_error() -> None:
    settings = replace(Settings.from_env(), max_upload_mb=1)
    store = FakeStore()
    service = RAGService(
        FakeEmbeddings(),
        FakeChat(),
        store,
        chunk_size=50,
        chunk_overlap=5,
        top_k=4,
        score_threshold=0.25,
    )

    with TestClient(create_app(settings, service)) as client:
        response = client.post(
            "/api/documents",
            files={"file": ("notes.docx", b"content", "application/octet-stream")},
        )

    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]
