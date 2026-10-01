import hashlib
from dataclasses import replace
from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient

from local_rag.api import create_app
from local_rag.config import Settings
from local_rag.domain import Chunk, DocumentInfo, GroundedClaim, GroundedResponse, SearchResult
from local_rag.service import RAGService


class FakeEmbeddings:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(len(text))] for text in texts]


class FakeChat:
    async def answer(self, question: str, context: str, citation_count: int) -> GroundedResponse:
        assert question and "notes.txt" in context and citation_count == 1
        return GroundedResponse(
            claims=[
                GroundedClaim(
                    text="**The evidence stays local**.\n\n"
                    "| Location | Access |\n|---|---|\n| Local | Private |\n\n"
                    "<script>alert('unsafe')</script>",
                    citations=[1],
                )
            ]
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
            chunks[0].page_count or 1,
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
        if "flow" in query:
            return [
                SearchResult(
                    "notes.txt",
                    1,
                    "```\nCollect evidence\n  ↓\nExtract facts\n  ↓\nValidate\n"
                    "  ├─ support\n  ├─ answerability\n  └─ <script>alert('proof')</script>",
                    0.94,
                    document_id,
                )
            ]
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


def test_web_api_flow(tmp_path: Path) -> None:
    settings = replace(Settings.from_env(), max_upload_mb=1, data_dir=tmp_path)
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
        assert index.headers["cache-control"] == "no-store"
        assert "/assets/app.js" in index.text
        script = client.get("/assets/app.js")
        assert script.status_code == 200
        assert "answer.innerHTML = data.answer_html" in script.text
        assert "text.innerHTML = item.text_html" in script.text
        assert client.get("/assets/styles.css").status_code == 200
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
        assert uploaded["original_available"]
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

        flow = client.post("/api/retrieve", json={"question": "flow diagram"}).json()[0]
        assert flow["text"].startswith("```")
        assert '<div class="evidence-flow"><ol>' in flow["text_html"]
        assert "<li>Validate" in flow["text_html"]
        assert "<li>answerability</li>" in flow["text_html"]
        assert "<script>" not in flow["text_html"]
        assert "&lt;script&gt;" in flow["text_html"]
        assert "```" not in flow["text_html"]

        assert client.delete(f"/api/documents/{uploaded['document_id']}").status_code == 200
        assert client.delete(f"/api/documents/{uploaded['document_id']}").status_code == 404
        assert client.delete("/api/documents").status_code == 200
        assert store.cleared

    assert store.closed


def test_upload_validation_returns_client_error(tmp_path: Path) -> None:
    settings = replace(Settings.from_env(), max_upload_mb=1, data_dir=tmp_path)
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


def test_original_pdf_preview_replacement_and_removal(tmp_path: Path) -> None:
    settings = replace(Settings.from_env(), data_dir=tmp_path)
    store = FakeStore()
    service = RAGService(
        FakeEmbeddings(),
        FakeChat(),
        store,
        chunk_size=200,
        chunk_overlap=20,
        top_k=4,
        score_threshold=0.25,
    )
    with pymupdf.open() as pdf:  # type: ignore[no-untyped-call]
        pdf.new_page()
        pdf.new_page().insert_text((72, 72), "Evidence on the second page")
        pdf.new_page()
        content = pdf.tobytes()

    with TestClient(create_app(settings, service)) as client:
        uploaded = client.post(
            "/api/documents", files={"file": ("report.pdf", content, "application/pdf")}
        ).json()
        document_id = uploaded["document_id"]
        assert uploaded["original_available"]
        assert client.get("/api/documents").json()[0]["pages"] == 3
        original = client.get(f"/api/documents/{document_id}/original")
        assert original.content == content
        assert "attachment" in original.headers["content-disposition"]
        assert original.headers["cache-control"] == "no-store"
        page = client.get(f"/api/documents/{document_id}/pages/2")
        assert page.status_code == 200
        assert page.headers["content-type"] == "image/png"
        assert page.content.startswith(b"\x89PNG")
        assert client.get(f"/api/documents/{document_id}/pages/1").status_code == 200
        assert client.get(f"/api/documents/{document_id}/pages/4").status_code == 400
        assert client.get(f"/api/documents/{document_id}/pages/0").status_code == 400
        assert client.get("/api/documents/unknown/original").status_code == 404
        assert client.get(f"/api/documents/{'0' * 64}/original").status_code == 404

        failed = client.post("/api/documents", files={"file": ("broken.pdf", b"broken")})
        assert failed.status_code == 400
        assert [path.name for path in (tmp_path / "sources").rglob("*") if path.is_file()] == [
            document_id
        ]

        with pymupdf.open() as pdf:  # type: ignore[no-untyped-call]
            pdf.new_page().insert_text((72, 72), "Updated evidence")
            replacement = pdf.tobytes()
        updated = client.post("/api/documents", files={"file": ("report.pdf", replacement)}).json()
        assert updated["document_id"] != document_id
        assert client.get(f"/api/documents/{document_id}/original").status_code == 404
        assert not list((tmp_path / "sources").rglob(document_id))
        assert client.delete(f"/api/documents/{updated['document_id']}").status_code == 200
        assert not [path for path in (tmp_path / "sources").rglob("*") if path.is_file()]

        client.post("/api/documents", files={"file": ("report.pdf", content)})
        assert client.delete("/api/documents").status_code == 200
        assert not [path for path in (tmp_path / "sources").rglob("*") if path.is_file()]


def test_legacy_documents_remain_usable_without_original_files(tmp_path: Path) -> None:
    settings = replace(Settings.from_env(), data_dir=tmp_path)
    store = FakeStore()
    store.document = DocumentInfo("legacy-id", "report.pdf", 1, 1)
    service = RAGService(
        FakeEmbeddings(),
        FakeChat(),
        store,
        chunk_size=200,
        chunk_overlap=20,
        top_k=4,
        score_threshold=0.25,
    )
    with TestClient(create_app(settings, service)) as client:
        assert not client.get("/api/documents").json()[0]["original_available"]
        assert client.get("/api/documents/legacy-id/original").status_code == 404
        assert client.delete("/api/documents/legacy-id").status_code == 200
