import hashlib
from dataclasses import replace

from fastapi.testclient import TestClient

from local_rag.api import create_app
from local_rag.config import Settings
from local_rag.domain import Chunk, DocumentInfo, SearchResult
from local_rag.service import RAGService


class FakeEmbeddings:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(len(text))] for text in texts]


class FakeChat:
    async def answer(self, question: str, context: str) -> str:
        assert question and "notes.txt" in context
        return (
            "**The evidence stays local** [1].\n\n"
            "| Location | Access |\n|---|---|\n| Local | Private [1] |\n\n"
            "<script>alert('unsafe')</script> [1]"
        )


class FakeStore:
    def __init__(self) -> None:
        self.closed = False
        self.cleared = False
        self.document: DocumentInfo | None = None

    def replace(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        assert chunks and vectors
        self.document = DocumentInfo(
            chunks[0].document_id,
            chunks[0].source,
            len(chunks),
            1,
            chunks[0].source_sha256,
        )

    def search(
        self,
        vector: list[float],
        query: str,
        limit: int,
        threshold: float,
        document_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        document_id = self.document.document_id if self.document else ""
        return [
            SearchResult(
                "notes.txt",
                1,
                "**Evidence stays local.**\n\n<script>alert('proof')</script>",
                0.92,
                document_id,
            )
        ]

    def list_documents(self) -> list[DocumentInfo]:
        return [self.document] if self.document else []

    def delete_document(self, document_id: str) -> bool:
        if self.document is None or self.document.document_id != document_id:
            return False
        self.document = None
        return True

    def clear(self) -> None:
        self.cleared = True
        self.document = None

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
        index = client.get("/")
        assert index.status_code == 200
        assert index.headers["x-frame-options"] == "DENY"
        assert "answer.innerHTML = data.answer_html" in index.text
        assert "text.innerHTML = item.text_html" in index.text
        assert "text.textContent = item.text" not in index.text
        assert client.get("/health").json()["chat_provider"] == "ollama"

        upload = client.post(
            "/api/documents",
            files={"file": ("notes.txt", b"Evidence stays local.", "text/plain")},
        )
        assert upload.status_code == 200
        assert upload.headers["cache-control"] == "no-store"
        uploaded = upload.json()
        assert uploaded["filename"] == "notes.txt"
        assert uploaded["source_sha256"] == hashlib.sha256(b"Evidence stays local.").hexdigest()
        assert uploaded["chunks"] == 1
        assert uploaded["pages"] == 1
        assert client.get("/api/documents").json() == [uploaded]

        query = client.post(
            "/api/query",
            json={
                "question": "Where is the evidence?",
                "document_ids": [uploaded["document_id"]],
            },
        )
        assert query.status_code == 200
        response = query.json()
        assert response["citations"][0]["score"] == 0.92
        assert response["citations"][0]["text"].startswith("**Evidence stays local.**")
        assert "<strong>Evidence stays local.</strong>" in response["citations"][0]["text_html"]
        assert "<script>" not in response["citations"][0]["text_html"]
        assert "&lt;script&gt;" in response["citations"][0]["text_html"]
        assert "<strong>The evidence stays local</strong>" in response["answer_html"]
        assert "<table>" in response["answer_html"]
        assert "<script>" not in response["answer_html"]
        assert "&lt;script&gt;" in response["answer_html"]

        retrieval = client.post("/api/retrieve", json={"question": "Evidence?"})
        assert retrieval.status_code == 200
        assert retrieval.json()[0]["document_id"] == uploaded["document_id"]
        assert "<strong>Evidence stays local.</strong>" in retrieval.json()[0]["text_html"]

        assert client.delete(f"/api/documents/{uploaded['document_id']}").status_code == 200
        assert client.delete(f"/api/documents/{uploaded['document_id']}").status_code == 404
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
