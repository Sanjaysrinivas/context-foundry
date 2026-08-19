"""Document loading and deterministic text chunking."""

from __future__ import annotations

import hashlib
import io
import re
import uuid
from pathlib import Path

from pypdf import PdfReader

from local_rag.domain import Chunk, Page, RAGError

SUPPORTED_EXTENSIONS = {".md", ".pdf", ".txt"}


def load_document(filename: str, content: bytes) -> list[Page]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise RAGError(f"Unsupported file type. Use one of: {supported}")

    if suffix == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(content))
            pages = [
                Page(filename, number, page.extract_text() or "")
                for number, page in enumerate(reader.pages, 1)
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
    for page in pages:
        document_hash.update(page.text.encode())
    document_id = document_hash.hexdigest()

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    for page in pages:
        text = re.sub(r"\s+", " ", page.text).strip()
        for index, start in enumerate(range(0, len(text), step)):
            excerpt = text[start : start + chunk_size].strip()
            if not excerpt:
                continue
            stable_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{page.number}:{index}:{excerpt}")
            )
            chunks.append(Chunk(stable_id, document_id, page.source, page.number, index, excerpt))
            if start + chunk_size >= len(text):
                break
    return chunks
