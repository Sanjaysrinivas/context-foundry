"""Run a human-verified golden dataset against the live local API."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import httpx

SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
ABSTENTION_PHRASES = (
    "could not find",
    "insufficient evidence",
    "not enough relevant evidence",
)


@dataclass(frozen=True, slots=True)
class ExpectedEvidence:
    source: str
    page: int
    contains: tuple[str, ...]
    source_sha256: str = ""
    evidence_group: str = "A"
    bbox: tuple[float, float, float, float] | None = None
    evidence_hash: str = ""


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    id: str
    question: str
    answerable: bool
    expected_evidence: tuple[ExpectedEvidence, ...]
    required_facts: tuple[str, ...]
    reference_answer: str | None = None
    document_ids: tuple[str, ...] = ()
    dataset_version: str = "1.0.0"
    split: str = "dev"
    corpus_id: str = "default"
    corpus_version: str = "unversioned"
    question_type: str = "direct_fact"
    generation_method: str = "human"
    review_status: str = "approved_gold"
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    cases: int
    hit_at_k: float
    mean_reciprocal_rank: float
    evidence_recall_at_k: float
    required_evidence_coverage_at_k: float
    ndcg_at_k: float
    citation_precision: float
    fact_coverage: float
    evidence_support: float
    abstention_accuracy: float
    false_answer_rate: float
    false_abstention_rate: float
    p50_latency_seconds: float
    p95_latency_seconds: float


def load_cases(path: Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise TypeError("case must be a JSON object")
            item = cast(dict[str, Any], raw)
            answerable = item.get("answerable")
            if not isinstance(answerable, bool):
                raise TypeError("answerable must be a boolean")
            case_id = item.get("case_id", item.get("id"))
            question = item.get("question")
            if not isinstance(case_id, str) or not isinstance(question, str):
                raise TypeError("case_id/id and question must be strings")
            reference_answer = item.get("reference_answer")
            if reference_answer is not None and not isinstance(reference_answer, str):
                raise TypeError("reference_answer must be a string or null")
            case = EvaluationCase(
                id=case_id,
                question=question,
                answerable=answerable,
                expected_evidence=tuple(
                    _load_evidence(value) for value in _list(item.get("expected_evidence", []))
                ),
                required_facts=_strings(item.get("required_facts", []), "required_facts"),
                reference_answer=reference_answer,
                document_ids=_strings(item.get("document_ids", []), "document_ids"),
                dataset_version=_string(item.get("dataset_version", "1.0.0"), "dataset_version"),
                split=_string(item.get("split", "dev"), "split"),
                corpus_id=_string(item.get("corpus_id", "default"), "corpus_id"),
                corpus_version=_string(item.get("corpus_version", "unversioned"), "corpus_version"),
                question_type=_string(
                    item.get("question_type", "direct_fact" if answerable else "unanswerable"),
                    "question_type",
                ),
                generation_method=_string(
                    item.get("generation_method", "human"), "generation_method"
                ),
                review_status=_string(item.get("review_status", "approved_gold"), "review_status"),
                tags=_strings(item.get("tags", []), "tags"),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid evaluation case on line {line_number}: {exc}") from exc

        if not case.id or not case.question:
            raise ValueError(f"Invalid evaluation case on line {line_number}: empty id or question")
        if case.id in seen_ids:
            raise ValueError(
                f"Invalid evaluation case on line {line_number}: duplicate id {case.id}"
            )
        if case.split not in {"dev", "holdout", "smoke"}:
            raise ValueError(
                f"Invalid evaluation case on line {line_number}: "
                "split must be dev, holdout, or smoke"
            )
        if not case.dataset_version or not case.corpus_id or not case.corpus_version:
            raise ValueError(
                f"Invalid evaluation case on line {line_number}: "
                "dataset and corpus versions must not be empty"
            )
        if case.review_status != "approved_gold":
            raise ValueError(
                f"Invalid evaluation case on line {line_number}: "
                "only approved_gold cases may enter release metrics"
            )
        if case.answerable and (
            not case.expected_evidence
            or not case.required_facts
            or any(not evidence.contains for evidence in case.expected_evidence)
        ):
            raise ValueError(
                f"Invalid evaluation case on line {line_number}: answerable cases require "
                "expected evidence with stable text and required facts"
            )
        if not case.answerable and (case.expected_evidence or case.required_facts):
            raise ValueError(
                f"Invalid evaluation case on line {line_number}: "
                "unanswerable cases cannot define expected evidence or required facts"
            )
        seen_ids.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError("Evaluation dataset is empty")
    return cases


async def run_evaluation(
    cases: list[EvaluationCase], base_url: str, timeout: float
) -> EvaluationMetrics:
    responses, latencies = await query_cases(cases, base_url, timeout)
    return calculate_metrics(cases, responses, latencies)


async def query_cases(
    cases: list[EvaluationCase], base_url: str, timeout: float
) -> tuple[list[dict[str, Any]], list[float]]:
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
    return responses, latencies


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
    expected_evidence_count = 0
    matched_evidence_count = 0
    required_group_count = 0
    matched_group_count = 0
    normalized_discounted_gain = 0.0
    relevant_citations = 0
    citation_count = 0
    required_fact_count = 0
    answer_fact_count = 0
    supported_fact_count = 0
    unanswerable = 0
    correct_abstentions = 0
    false_abstentions = 0

    for case, response in zip(cases, responses, strict=True):
        citations = cast(list[dict[str, Any]], response.get("citations", []))
        answer = str(response.get("answer", ""))
        citation_count += len(citations)
        relevant_citations += sum(
            any(_matches(citation, expected) for expected in case.expected_evidence)
            for citation in citations
        )

        if not case.answerable:
            unanswerable += 1
            if not citations and _is_abstention(answer):
                correct_abstentions += 1
            continue

        answerable += 1
        if not citations or _is_abstention(answer):
            false_abstentions += 1
        ranks = [
            rank
            for rank, citation in enumerate(citations, 1)
            if any(_matches(citation, expected) for expected in case.expected_evidence)
        ]
        if ranks:
            hits += 1
            reciprocal_rank += 1 / ranks[0]

        matched_evidence = {
            index
            for index, expected in enumerate(case.expected_evidence)
            if any(_matches(citation, expected) for citation in citations)
        }
        expected_evidence_count += len(case.expected_evidence)
        matched_evidence_count += len(matched_evidence)

        groups = {evidence.evidence_group for evidence in case.expected_evidence}
        matched_groups = {
            evidence.evidence_group
            for evidence in case.expected_evidence
            if any(_matches(citation, evidence) for citation in citations)
        }
        required_group_count += len(groups)
        matched_group_count += len(matched_groups)
        normalized_discounted_gain += _ndcg(citations, case.expected_evidence, len(groups))

        evidence_text = " ".join(str(citation.get("text", "")) for citation in citations)
        required_fact_count += len(case.required_facts)
        answer_fact_count += sum(_contains(answer, fact) for fact in case.required_facts)
        supported_fact_count += sum(_contains(evidence_text, fact) for fact in case.required_facts)

    abstention_accuracy = _ratio(correct_abstentions, unanswerable)
    return EvaluationMetrics(
        cases=len(cases),
        hit_at_k=_ratio(hits, answerable),
        mean_reciprocal_rank=_ratio(reciprocal_rank, answerable),
        evidence_recall_at_k=_ratio(matched_evidence_count, expected_evidence_count),
        required_evidence_coverage_at_k=_ratio(matched_group_count, required_group_count),
        ndcg_at_k=_ratio(normalized_discounted_gain, answerable),
        citation_precision=_ratio(relevant_citations, citation_count),
        fact_coverage=_ratio(answer_fact_count, required_fact_count),
        evidence_support=_ratio(supported_fact_count, required_fact_count),
        abstention_accuracy=abstention_accuracy,
        false_answer_rate=1.0 - abstention_accuracy,
        false_abstention_rate=_ratio(false_abstentions, answerable),
        p50_latency_seconds=float(statistics.median(latencies)) if latencies else 0.0,
        p95_latency_seconds=_percentile_95(latencies),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--min-hit-at-k", type=float, default=0.85)
    parser.add_argument("--min-mrr", type=float, default=0.70)
    parser.add_argument("--min-evidence-recall", type=float, default=0.90)
    parser.add_argument("--min-required-evidence-coverage", type=float, default=0.85)
    parser.add_argument("--min-ndcg", type=float, default=0.70)
    parser.add_argument("--min-citation-precision", type=float, default=0.90)
    parser.add_argument("--min-fact-coverage", type=float, default=0.90)
    parser.add_argument("--min-evidence-support", type=float, default=0.90)
    parser.add_argument("--min-abstention", type=float, default=0.90)
    parser.add_argument("--max-false-abstention", type=float, default=0.10)
    args = parser.parse_args()
    metrics = asyncio.run(run_evaluation(load_cases(args.dataset), args.base_url, args.timeout))
    print(json.dumps(asdict(metrics), indent=2))
    passed = (
        metrics.hit_at_k >= args.min_hit_at_k
        and metrics.mean_reciprocal_rank >= args.min_mrr
        and metrics.evidence_recall_at_k >= args.min_evidence_recall
        and metrics.required_evidence_coverage_at_k >= args.min_required_evidence_coverage
        and metrics.ndcg_at_k >= args.min_ndcg
        and metrics.citation_precision >= args.min_citation_precision
        and metrics.fact_coverage >= args.min_fact_coverage
        and metrics.evidence_support >= args.min_evidence_support
        and metrics.abstention_accuracy >= args.min_abstention
        and metrics.false_abstention_rate <= args.max_false_abstention
    )
    raise SystemExit(0 if passed else 1)


def _load_evidence(value: object) -> ExpectedEvidence:
    if not isinstance(value, dict):
        raise TypeError("expected_evidence entries must be objects")
    entry = cast(dict[str, Any], value)
    source = _string(entry.get("source", ""), "expected_evidence.source")
    page = entry.get("page")
    if not isinstance(page, int) or isinstance(page, bool) or page < 1:
        raise TypeError("expected_evidence.page must be a positive integer")
    source_sha256 = _string(entry.get("source_sha256", ""), "source_sha256")
    if source_sha256 and not SHA256_RE.fullmatch(source_sha256):
        raise ValueError("source_sha256 must contain 64 hexadecimal characters")
    if not source and not source_sha256:
        raise ValueError("expected evidence requires source or source_sha256")
    evidence_group = _string(entry.get("evidence_group", "A"), "evidence_group")
    if not evidence_group:
        raise ValueError("evidence_group must not be empty")
    return ExpectedEvidence(
        source=source,
        page=page,
        contains=_strings(entry.get("contains", []), "expected_evidence.contains"),
        source_sha256=source_sha256.lower(),
        evidence_group=evidence_group,
        bbox=_bbox(entry.get("bbox")),
        evidence_hash=_string(entry.get("evidence_hash", ""), "evidence_hash"),
    )


def _matches(citation: dict[str, Any], expected: ExpectedEvidence) -> bool:
    if expected.source_sha256:
        source_matches = (
            str(citation.get("source_sha256", "")).casefold() == expected.source_sha256.casefold()
        )
    else:
        source_matches = str(citation.get("source", "")).casefold() == expected.source.casefold()
    return (
        source_matches
        and int(citation.get("page", 0)) == expected.page
        and all(_contains(str(citation.get("text", "")), value) for value in expected.contains)
    )


def _ndcg(
    citations: list[dict[str, Any]], evidence: tuple[ExpectedEvidence, ...], group_count: int
) -> float:
    seen_groups: set[str] = set()
    discounted_gain = 0.0
    for rank, citation in enumerate(citations, 1):
        matching_groups = {
            expected.evidence_group for expected in evidence if _matches(citation, expected)
        }
        if matching_groups - seen_groups:
            discounted_gain += 1 / math.log2(rank + 1)
            seen_groups.update(matching_groups)
    ideal_count = min(group_count, len(citations))
    ideal_gain = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return discounted_gain / ideal_gain if ideal_gain else 1.0


def _contains(text: str, value: str) -> bool:
    return " ".join(value.casefold().split()) in " ".join(text.casefold().split())


def _is_abstention(answer: str) -> bool:
    normalized = answer.casefold()
    return any(phrase in normalized for phrase in ABSTENTION_PHRASES)


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError("expected_evidence must be an array")
    return cast(list[object], value)


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{name} must be an array of strings")
    return tuple(cast(list[str], value))


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _bbox(value: object) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise TypeError("bbox must contain four numbers")
    try:
        coordinates = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise TypeError("bbox must contain four numbers") from exc
    bbox = cast(tuple[float, float, float, float], coordinates)
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ValueError("bbox must have positive width and height")
    return bbox


def _ratio(numerator: float, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


if __name__ == "__main__":
    main()
