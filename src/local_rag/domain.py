"""Small domain types shared by the RAG pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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


AuditRelation = Literal["supported", "contradicted", "insufficient"]
AuditAction = Literal["keep", "withhold", "review", "unavailable"]


class AuditProbabilities(BaseModel):
    """The three mutually exclusive citation relations returned by Jev."""

    model_config = ConfigDict(extra="forbid", strict=True)

    supported: float = Field(ge=0, le=1, allow_inf_nan=False)
    contradicted: float = Field(ge=0, le=1, allow_inf_nan=False)
    insufficient: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def require_distribution(self) -> AuditProbabilities:
        if not math.isclose(sum(self.model_dump().values()), 1.0, abs_tol=1e-5):
            raise ValueError("citation probabilities must sum to one")
        return self


class CitationAssessment(BaseModel):
    """A validated Choice answer; confidence is separate from support."""

    model_config = ConfigDict(strict=True)

    type: Literal["choice"]
    choice: AuditRelation
    probabilities: AuditProbabilities
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def require_winning_choice(self) -> CitationAssessment:
        values = self.probabilities.model_dump()
        if values[self.choice] < max(values.values()) - 1e-6:
            raise ValueError("choice must have the highest probability")
        return self


class CitationCheck(BaseModel):
    """A citation assessment and the exact evidence submitted for it."""

    citations: list[int]
    evidence_hashes: dict[int, str]
    status: Literal["completed", "unavailable"]
    suggested_action: AuditAction
    threshold: float
    requested_model: str
    request_id: str
    elapsed_seconds: float
    assessment: CitationAssessment | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None
    policy_version: str = "citation-v1"
    prompt_version: str = "citation-v1"


class ClaimAudit(CitationCheck):
    """Final claim check, retaining evidence from any citation repair attempts."""

    claim_id: int
    text: str
    original: CitationCheck | None = None
    repair_attempts: list[CitationCheck] = Field(default_factory=list)
    repair_policy_version: str | None = None


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
    audits: list[ClaimAudit] = field(default_factory=list)
