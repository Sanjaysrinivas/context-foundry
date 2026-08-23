"""Document loading and deterministic text chunking."""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import Any, cast

import pymupdf
import pymupdf4llm  # type: ignore[import-untyped]
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from local_rag.domain import Chunk, Page, RAGError

SUPPORTED_EXTENSIONS = {".md", ".pdf", ".txt"}
MARK_TAG_RE = re.compile(r"</?mark>", re.IGNORECASE)
CODE_PLURAL_RE = re.compile(r"`([^`]+)`\s+s\b")
PUNCTUATION_SPACE_RE = re.compile(r"\s+([.,;:])")
SLASH_SPACE_RE = re.compile(r"(?<=\S)/\s+")
FLOW_BRANCH_RE = re.compile(r"^[├└][─-]\s*(.+)$")


def clean_extracted_markdown(text: str) -> str:
    text = MARK_TAG_RE.sub("", text)
    text = CODE_PLURAL_RE.sub(lambda match: f"`{match.group(1)}s`", text)
    text = PUNCTUATION_SPACE_RE.sub(r"\1", text)
    return SLASH_SPACE_RE.sub("/", text)


def flow_diagram_markdown(text: str) -> str | None:
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

    output: list[str] = []
    for index, (label, details) in enumerate(stages, 1):
        output.append(f"{index}. {label}")
        output.extend(f"   - {detail}" for detail in details)
    return "\n".join(output)


def load_document(filename: str, content: bytes) -> list[Page]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise RAGError(f"Unsupported file type. Use one of: {supported}")

    source_sha256 = hashlib.sha256(content).hexdigest()
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
                Page(
                    filename,
                    int(page["metadata"]["page_number"]),
                    clean_extracted_markdown(str(page["text"])),
                    source_sha256,
                )
                for page in extracted
            ]
        except Exception as exc:
            raise RAGError("The PDF could not be read") from exc
    else:
        pages = [Page(filename, 1, content.decode("utf-8", errors="replace"), source_sha256)]

    if not any(page.text.strip() for page in pages):
        raise RAGError("The document contains no extractable text")
    return pages


def chunk_pages(pages: list[Page], chunk_size: int, overlap: int) -> list[Chunk]:
    document_id = pages[0].source_sha256 if pages and pages[0].source_sha256 else ""
    if not document_id:
        document_hash = hashlib.sha256()
        if pages:
            document_hash.update(pages[0].source.casefold().encode())
        for page in pages:
            document_hash.update(page.text.encode())
        document_id = document_hash.hexdigest()

    chunks: list[Chunk] = []
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    for page in pages:
        for index, text in enumerate(text_splitter.split_text(page.text)):
            excerpt = text.strip()
            stable_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{page.number}:{index}:{excerpt}")
            )
            chunks.append(
                Chunk(
                    stable_id,
                    document_id,
                    page.source,
                    page.number,
                    index,
                    excerpt,
                    page.source_sha256,
                )
            )
    return chunks
