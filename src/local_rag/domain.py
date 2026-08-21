"""Small domain types shared by the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass


class RAGError(Exception):
    """Base class for expected application errors."""


class ProviderError(RAGError):
    """A model provider could not satisfy a request."""


@dataclass(frozen=True, slots=True)
class Page:
    source: str
    number: int
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    document_id: str
    source: str
    page: int
    index: int
    text: str


@dataclass(frozen=True, slots=True)
class SearchResult:
    source: str
    page: int
    text: str
    score: float
    document_id: str = ""


@dataclass(frozen=True, slots=True)
class DocumentInfo:
    document_id: str
    source: str
    chunks: int
    pages: int


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    citations: list[SearchResult]
