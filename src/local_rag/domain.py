"""Small domain types shared by the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RAGError(Exception):
    """Base class for expected application errors."""


class ProviderError(RAGError):
    """A model provider could not satisfy a request."""


class GroundedClaim(BaseModel):
    """One model-generated claim and the retrieved excerpts that support it."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=2_000)
    citations: list[Annotated[int, Field(ge=1)]] = Field(min_length=1)


class GroundedResponse(BaseModel):
    """Schema-constrained generation result rendered by the application."""

    model_config = ConfigDict(extra="forbid")

    style: Literal["paragraphs", "bullets", "steps"] = "paragraphs"
    claims: list[GroundedClaim] = Field(default_factory=list, max_length=50)
    unsupported: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def require_an_outcome(self) -> GroundedResponse:
        if not self.claims and not self.unsupported:
            raise ValueError("claims or unsupported must contain an answer outcome")
        return self


@dataclass(frozen=True, slots=True)
class Page:
    source: str
    number: int
    text: str
    source_sha256: str = ""


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    document_id: str
    source: str
    page: int
    index: int
    text: str
    source_sha256: str = ""


@dataclass(frozen=True, slots=True)
class SearchResult:
    source: str
    page: int
    text: str
    score: float
    document_id: str = ""
    source_sha256: str = ""


@dataclass(frozen=True, slots=True)
class DocumentInfo:
    document_id: str
    source: str
    chunks: int
    pages: int
    source_sha256: str = ""


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    citations: list[SearchResult]
