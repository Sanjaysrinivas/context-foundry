"""Document loading and deterministic text chunking."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any, cast

import pymupdf
import pymupdf4llm  # type: ignore[import-untyped]
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from local_rag.domain import Chunk, Page, RAGError

SUPPORTED_EXTENSIONS = {".md", ".pdf", ".txt"}


def load_document(filename: str, content: bytes) -> list[Page]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise RAGError(f"Unsupported file type. Use one of: {supported}")

    if suffix == ".pdf":
        try:
            with pymupdf.open(  # type: ignore[no-untyped-call]
                stream=content, filetype="pdf"
            ) as document:
                extracted = cast(
                    list[dict[str, Any]],
                    pymupdf4llm.to_markdown(document, page_chunks=True, use_ocr=True),
                )
            pages = [
                Page(filename, int(page["metadata"]["page_number"]), str(page["text"]))
                for page in extracted
            ]
        except Exception as exc:
            raise RAGError("The PDF could not be read") from exc
    else:
        pages = [Page(filename, 1, content.decode("utf-8", errors="replace"))]

    if not any(page.text.strip() for page in pages):
        raise RAGError("The document contains no extractable text")
    return pages


def chunk_pages(pages: list[Page], chunk_size: int, overlap: int) -> list[Chunk]:
    document_hash = hashlib.sha256()
    if pages:
        document_hash.update(pages[0].source.casefold().encode())
    for page in pages:
        document_hash.update(page.text.encode())
    document_id = document_hash.hexdigest()

    chunks: list[Chunk] = []
    markdown_splitter = MarkdownHeaderTextSplitter(
        [("#", "title"), ("##", "section"), ("###", "subsection")],
        strip_headers=False,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    for page in pages:
        sections = markdown_splitter.split_text(page.text)
        excerpts = text_splitter.split_documents(sections)
        for index, document in enumerate(excerpts):
            excerpt = document.page_content.strip()
            stable_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{page.number}:{index}:{excerpt}")
            )
            chunks.append(Chunk(stable_id, document_id, page.source, page.number, index, excerpt))
    return chunks
