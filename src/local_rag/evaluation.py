"""Run a private golden dataset against the live local API."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import httpx


@dataclass(frozen=True, slots=True)
class ExpectedEvidence:
    source: str
    page: int
    contains: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    id: str
    question: str
    answerable: bool
    expected_evidence: tuple[ExpectedEvidence, ...]
    required_facts: tuple[str, ...]
    reference_answer: str | None = None
    document_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    cases: int
    hit_at_k: float
    mean_reciprocal_rank: float
    citation_precision: float
    fact_coverage: float
    evidence_support: float
    abstention_accuracy: float
    p95_latency_seconds: float


def load_cases(path: Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = cast(dict[str, Any], json.loads(line))
            if not isinstance(item.get("answerable"), bool):
                raise TypeError("answerable must be a boolean")
            evidence = tuple(
                ExpectedEvidence(
                    source=str(entry["source"]),
                    page=int(entry["page"]),
                    contains=tuple(str(value) for value in entry.get("contains", [])),
                )
                for entry in item.get("expected_evidence", [])
            )
            case = EvaluationCase(
                id=str(item["id"]),
                question=str(item["question"]),
                answerable=bool(item["answerable"]),
                expected_evidence=evidence,
                required_facts=tuple(str(value) for value in item.get("required_facts", [])),
                reference_answer=(
                    str(item["reference_answer"])
                    if item.get("reference_answer") is not None
                    else None
                ),
                document_ids=tuple(str(value) for value in item.get("document_ids", [])),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid evaluation case on line {line_number}: {exc}") from exc
        if (
            not case.id
            or not case.question
            or (case.answerable and (not case.expected_evidence or not case.required_facts))
        ):
            raise ValueError(
                f"Invalid evaluation case on line {line_number}: "
                "answerable cases require id, question, expected evidence, and required facts"
            )
        cases.append(case)
    if not cases:
        raise ValueError("Evaluation dataset is empty")
    return cases


async def run_evaluation(
    cases: list[EvaluationCase], base_url: str, timeout: float
) -> EvaluationMetrics:
    responses: list[dict[str, Any]] = []
    latencies: list[float] = []
    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout) as client:
        for case in cases:
            started = time.perf_counter()
            response = await client.post(
                "/api/query",
                json={"question": case.question, "document_ids": list(case.document_ids)},
            )
            response.raise_for_status()
            latencies.append(time.perf_counter() - started)
            responses.append(cast(dict[str, Any], response.json()))
    return calculate_metrics(cases, responses, latencies)


def calculate_metrics(
    cases: list[EvaluationCase],
    responses: list[dict[str, Any]],
    latencies: list[float],
) -> EvaluationMetrics:
    if len(cases) != len(responses) or len(cases) != len(latencies):
        raise ValueError("Cases, responses, and latencies must have equal lengths")

    answerable = 0
    hits = 0
    reciprocal_rank = 0.0
    relevant_citations = 0
    citation_count = 0
    required_fact_count = 0
    answer_fact_count = 0
    supported_fact_count = 0
    unanswerable = 0
    correct_abstentions = 0

    for case, response in zip(cases, responses, strict=True):
        citations = cast(list[dict[str, Any]], response.get("citations", []))
        answer = str(response.get("answer", ""))
        if not case.answerable:
            unanswerable += 1
            if not citations and "could not find" in answer.lower():
                correct_abstentions += 1
            continue

        answerable += 1
        ranks = [
            rank
            for rank, citation in enumerate(citations, 1)
            if any(_matches(citation, expected) for expected in case.expected_evidence)
        ]
        if ranks:
            hits += 1
            reciprocal_rank += 1 / ranks[0]
        citation_count += len(citations)
        relevant_citations += sum(
            any(_matches(citation, expected) for expected in case.expected_evidence)
            for citation in citations
        )

        evidence_text = " ".join(str(citation.get("text", "")) for citation in citations)
        required_fact_count += len(case.required_facts)
        answer_fact_count += sum(_contains(answer, fact) for fact in case.required_facts)
        supported_fact_count += sum(_contains(evidence_text, fact) for fact in case.required_facts)

    return EvaluationMetrics(
        cases=len(cases),
        hit_at_k=_ratio(hits, answerable),
        mean_reciprocal_rank=_ratio(reciprocal_rank, answerable),
        citation_precision=_ratio(relevant_citations, citation_count),
        fact_coverage=_ratio(answer_fact_count, required_fact_count),
        evidence_support=_ratio(supported_fact_count, required_fact_count),
        abstention_accuracy=_ratio(correct_abstentions, unanswerable),
        p95_latency_seconds=_percentile_95(latencies),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--min-hit-at-k", type=float, default=0.85)
    parser.add_argument("--min-mrr", type=float, default=0.70)
    parser.add_argument("--min-citation-precision", type=float, default=0.90)
    parser.add_argument("--min-fact-coverage", type=float, default=0.90)
    parser.add_argument("--min-evidence-support", type=float, default=0.90)
    parser.add_argument("--min-abstention", type=float, default=0.90)
    args = parser.parse_args()
    metrics = asyncio.run(run_evaluation(load_cases(args.dataset), args.base_url, args.timeout))
    print(json.dumps(asdict(metrics), indent=2))
    passed = (
        metrics.hit_at_k >= args.min_hit_at_k
        and metrics.mean_reciprocal_rank >= args.min_mrr
        and metrics.citation_precision >= args.min_citation_precision
        and metrics.fact_coverage >= args.min_fact_coverage
        and metrics.evidence_support >= args.min_evidence_support
        and metrics.abstention_accuracy >= args.min_abstention
    )
    raise SystemExit(0 if passed else 1)


def _matches(citation: dict[str, Any], expected: ExpectedEvidence) -> bool:
    return (
        str(citation.get("source", "")).casefold() == expected.source.casefold()
        and int(citation.get("page", 0)) == expected.page
        and all(_contains(str(citation.get("text", "")), value) for value in expected.contains)
    )


def _contains(text: str, value: str) -> bool:
    return " ".join(value.casefold().split()) in " ".join(text.casefold().split())


def _ratio(numerator: float, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


if __name__ == "__main__":
    main()
