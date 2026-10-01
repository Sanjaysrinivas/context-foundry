"""FastAPI entry point and browser-facing API."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Literal
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from markdown_it import MarkdownIt
from pydantic import BaseModel, Field

from local_rag.config import Settings
from local_rag.documents import flow_diagram_markdown, render_pdf_page
from local_rag.domain import ClaimAudit, DocumentInfo, ProviderError, RAGError, SearchResult
from local_rag.factory import build_service
from local_rag.service import RAGService

WEB_DIR = Path(__file__).parent / "web"
MARKDOWN = MarkdownIt("gfm-like", {"html": False, "linkify": False})


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    document_ids: list[str] = Field(default_factory=list, max_length=50)


class CitationResponse(BaseModel):
    document_id: str
    source: str
    page: int
    text: str
    text_html: str
    score: float
    source_sha256: str


class QueryResponse(BaseModel):
    answer: str
    answer_html: str
    citations: list[CitationResponse]
    audit_mode: Literal["off", "observe", "repair"] = "off"
    audits: list[ClaimAudit] = Field(default_factory=list)


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    chunks: int
    pages: int
    source_sha256: str
    original_available: bool = False


def _flow_diagram_html(text: str) -> str | None:
    diagram = flow_diagram_markdown(text)
    return f'<div class="evidence-flow">{MARKDOWN.render(diagram)}</div>' if diagram else None


def _citation_response(item: SearchResult) -> CitationResponse:
    return CitationResponse(
        document_id=item.document_id,
        source=item.source,
        page=item.page,
        text=item.text,
        text_html=_flow_diagram_html(item.text) or MARKDOWN.render(item.text),
        score=item.score,
        source_sha256=item.source_sha256,
    )


def create_app(settings: Settings | None = None, service: RAGService | None = None) -> FastAPI:
    configured = settings or Settings.from_env()
    rag = service
    sources_dir = (
        configured.data_dir / "sources" / hashlib.sha256(configured.collection.encode()).hexdigest()
    )

    def source_path(document_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{64}", document_id):
            raise HTTPException(status_code=404, detail="Document not found")
        return sources_dir / document_id

    def remove_source(document_id: str) -> None:
        if re.fullmatch(r"[a-f0-9]{64}", document_id):
            source_path(document_id).unlink(missing_ok=True)

    def document_response(document: DocumentInfo) -> DocumentResponse:
        return DocumentResponse(
            document_id=document.document_id,
            filename=document.source,
            chunks=document.chunks,
            pages=document.pages,
            source_sha256=document.source_sha256,
            original_available=bool(
                re.fullmatch(r"[a-f0-9]{64}", document.document_id)
                and source_path(document.document_id).is_file()
            ),
        )

    async def original_document(document_id: str) -> tuple[DocumentInfo, Path]:
        path = source_path(document_id)
        documents = await active_service().list_documents()
        document = next((item for item in documents if item.document_id == document_id), None)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        if not path.is_file():
            raise HTTPException(
                status_code=404,
                detail="Add this document again to view its original file.",
            )
        return document, path

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        nonlocal rag
        rag = rag or build_service(configured)
        yield
        rag.close()

    app = FastAPI(title="Context Foundry", version="0.1.0", lifespan=lifespan)
    app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")

    @app.middleware("http")
    async def security_headers(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; "
            "style-src 'self'; script-src 'self'; object-src 'none'; base-uri 'self'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path == "/" or request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def active_service() -> RAGService:
        if rag is None:
            raise RuntimeError("Application has not started")
        return rag

    @app.exception_handler(RAGError)
    async def handle_rag_error(_request: Request, exc: RAGError) -> JSONResponse:
        status = 503 if isinstance(exc, ProviderError) else 400
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/health")
    async def health() -> dict[str, str]:
        def local_provider(provider: str) -> bool:
            url = configured.ollama_base_url if provider == "ollama" else configured.openai_base_url
            return urlparse(url).hostname in {"localhost", "127.0.0.1", "::1"}

        return {
            "status": "ok",
            "chat_provider": configured.chat_provider,
            "embedding_provider": configured.embedding_provider,
            "jev_mode": configured.jev_mode,
            "max_upload_mb": str(configured.max_upload_mb),
            "local_models": str(
                local_provider(configured.chat_provider)
                and local_provider(configured.embedding_provider)
            ).lower(),
        }

    @app.post("/api/documents")
    async def ingest_document(
        file: Annotated[UploadFile, File()],
    ) -> DocumentResponse:
        filename = file.filename or "document"
        limit = configured.max_upload_mb * 1024 * 1024
        content = await file.read(limit + 1)
        if len(content) > limit:
            raise RAGError(f"File exceeds the {configured.max_upload_mb} MB upload limit")
        previous = await active_service().list_documents()
        # Stage the source before indexing so a full disk cannot silently discard the original.
        staged_path: Path | None = None
        try:
            sources_dir.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=sources_dir, delete=False) as staged:
                staged_path = Path(staged.name)
                staged.write(content)
            document = await active_service().ingest(filename, content)
            staged_path.replace(source_path(document.document_id))
        except OSError as exc:
            raise RAGError(
                "The original file could not be saved. "
                "Check write access and free disk space, then try again."
            ) from exc
        finally:
            if staged_path is not None:
                staged_path.unlink(missing_ok=True)
        for old in previous:
            if old.source == filename and old.document_id != document.document_id:
                remove_source(old.document_id)
        return document_response(document)

    @app.get("/api/documents", response_model=list[DocumentResponse])
    async def list_documents() -> list[DocumentResponse]:
        documents = await active_service().list_documents()
        return [document_response(document) for document in documents]

    @app.get("/api/documents/{document_id}/original")
    async def download_original(document_id: str) -> FileResponse:
        document, path = await original_document(document_id)
        return FileResponse(path, filename=document.source, media_type="application/octet-stream")

    @app.get("/api/documents/{document_id}/pages/{page_number}")
    async def preview_page(document_id: str, page_number: int) -> Response:
        document, path = await original_document(document_id)
        if Path(document.source).suffix.lower() != ".pdf":
            raise HTTPException(status_code=400, detail="Page previews are available for PDFs.")
        image = await asyncio.to_thread(render_pdf_page, path, page_number)
        return Response(image, media_type="image/png")

    @app.post("/api/query", response_model=QueryResponse)
    async def query(request: QueryRequest) -> QueryResponse:
        active = active_service()
        result = await active.ask(request.question, request.document_ids or None)
        return QueryResponse(
            answer=result.text,
            answer_html=MARKDOWN.render(result.text),
            citations=[_citation_response(item) for item in result.citations],
            audit_mode=("repair" if active.repair_citations else "observe")
            if active.auditor is not None
            else "off",
            audits=result.audits,
        )

    @app.post("/api/retrieve", response_model=list[CitationResponse])
    async def retrieve(request: QueryRequest) -> list[CitationResponse]:
        matches = await active_service().retrieve(request.question, request.document_ids or None)
        return [_citation_response(item) for item in matches]

    @app.delete("/api/documents/{document_id}")
    async def delete_document(document_id: str) -> dict[str, str]:
        if not await active_service().delete_document(document_id):
            raise HTTPException(status_code=404, detail="Document not found")
        remove_source(document_id)
        return {"status": "deleted"}

    @app.delete("/api/documents")
    async def clear_documents() -> dict[str, str]:
        documents = await active_service().list_documents()
        await active_service().clear()
        for document in documents:
            remove_source(document.document_id)
        return {"status": "cleared"}

    return app


app = create_app()


def main() -> None:
    uvicorn.run("local_rag.api:app", host="127.0.0.1", port=8000, reload=True)
