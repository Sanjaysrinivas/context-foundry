"""FastAPI entry point and browser-facing API."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from markdown_it import MarkdownIt
from pydantic import BaseModel, Field

from local_rag.config import Settings
from local_rag.domain import ProviderError, RAGError, SearchResult
from local_rag.factory import build_service
from local_rag.service import RAGService

WEB_DIR = Path(__file__).parent / "web"
MARKDOWN = MarkdownIt("gfm-like", {"html": False, "linkify": False})
FLOW_BRANCH_RE = re.compile(r"^[├└][─-]\s*(.+)$")


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


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    chunks: int
    pages: int
    source_sha256: str


def _flow_diagram_html(text: str) -> str | None:
    lines = text.strip().splitlines()
    if not lines or not lines[0].strip().startswith("```"):
        return None
    lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines.pop()
    if sum(line.strip() == "↓" for line in lines) < 2:
        return None

    stages: list[tuple[str, list[str]]] = []
    for line in lines:
        content = line.strip()
        if not content or content == "↓":
            continue
        branch = FLOW_BRANCH_RE.match(content)
        if branch:
            if not stages:
                return None
            stages[-1][1].append(branch.group(1))
        else:
            stages.append((content, []))
    if not stages or not any(details for _label, details in stages):
        return None

    items = []
    for label, details in stages:
        detail_list = ""
        if details:
            detail_list = (
                "<ul>" + "".join(f"<li>{escape(detail)}</li>" for detail in details) + "</ul>"
            )
        items.append(f"<li><span>{escape(label)}</span>{detail_list}</li>")
    return '<div class="evidence-flow"><ol>' + "".join(items) + "</ol></div>"


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

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        nonlocal rag
        rag = rag or build_service(configured)
        yield
        rag.close()

    app = FastAPI(title="Local RAG", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def security_headers(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; "
            "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'"
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
        return {
            "status": "ok",
            "chat_provider": configured.chat_provider,
            "embedding_provider": configured.embedding_provider,
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
        document = await active_service().ingest(filename, content)
        return DocumentResponse(
            document_id=document.document_id,
            filename=document.source,
            chunks=document.chunks,
            pages=document.pages,
            source_sha256=document.source_sha256,
        )

    @app.get("/api/documents", response_model=list[DocumentResponse])
    async def list_documents() -> list[DocumentResponse]:
        documents = await active_service().list_documents()
        return [
            DocumentResponse(
                document_id=document.document_id,
                filename=document.source,
                chunks=document.chunks,
                pages=document.pages,
                source_sha256=document.source_sha256,
            )
            for document in documents
        ]

    @app.post("/api/query", response_model=QueryResponse)
    async def query(request: QueryRequest) -> QueryResponse:
        result = await active_service().ask(request.question, request.document_ids or None)
        return QueryResponse(
            answer=result.text,
            answer_html=MARKDOWN.render(result.text),
            citations=[_citation_response(item) for item in result.citations],
        )

    @app.post("/api/retrieve", response_model=list[CitationResponse])
    async def retrieve(request: QueryRequest) -> list[CitationResponse]:
        matches = await active_service().retrieve(request.question, request.document_ids or None)
        return [_citation_response(item) for item in matches]

    @app.delete("/api/documents/{document_id}")
    async def delete_document(document_id: str) -> dict[str, str]:
        if not await active_service().delete_document(document_id):
            raise HTTPException(status_code=404, detail="Document not found")
        return {"status": "deleted"}

    @app.delete("/api/documents")
    async def clear_documents() -> dict[str, str]:
        await active_service().clear()
        return {"status": "cleared"}

    return app


app = create_app()


def main() -> None:
    uvicorn.run("local_rag.api:app", host="127.0.0.1", port=8000, reload=True)
