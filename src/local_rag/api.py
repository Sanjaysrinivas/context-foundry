"""FastAPI entry point and browser-facing API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from local_rag.config import Settings
from local_rag.domain import ProviderError, RAGError
from local_rag.factory import build_service
from local_rag.service import RAGService

WEB_DIR = Path(__file__).parent / "web"


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)


class CitationResponse(BaseModel):
    source: str
    page: int
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]


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
    ) -> dict[str, int | str]:
        filename = file.filename or "document"
        limit = configured.max_upload_mb * 1024 * 1024
        content = await file.read(limit + 1)
        if len(content) > limit:
            raise RAGError(f"File exceeds the {configured.max_upload_mb} MB upload limit")
        chunks = await active_service().ingest(filename, content)
        return {"filename": filename, "chunks": chunks}

    @app.post("/api/query", response_model=QueryResponse)
    async def query(request: QueryRequest) -> QueryResponse:
        result = await active_service().ask(request.question)
        return QueryResponse(
            answer=result.text,
            citations=[
                CitationResponse(
                    source=item.source,
                    page=item.page,
                    text=item.text,
                    score=item.score,
                )
                for item in result.citations
            ],
        )

    @app.delete("/api/documents")
    async def clear_documents() -> dict[str, str]:
        active_service().clear()
        return {"status": "cleared"}

    return app


app = create_app()


def main() -> None:
    uvicorn.run("local_rag.api:app", host="127.0.0.1", port=8000, reload=True)
