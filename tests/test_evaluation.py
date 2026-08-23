import json
from pathlib import Path

import pytest

from local_rag.evaluation import (
    EvaluationCase,
    ExpectedEvidence,
    calculate_metrics,
    load_cases,
)


@pytest.mark.unit
def test_loads_jsonl_and_calculates_quality_metrics(tmp_path: Path) -> None:
    source_sha256 = "a" * 64
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "case_id": "skills-1",
                "dataset_version": "1.0.0",
                "corpus_id": "portfolio",
                "corpus_version": "2026-08",
                "split": "holdout",
                "question": "What skills are listed?",
                "answerable": True,
                "question_type": "direct_fact",
                "expected_evidence": [
                    {
                        "source": "resume.pdf",
                        "source_sha256": source_sha256,
                        "page": 1,
                        "contains": ["Python"],
                        "evidence_group": "skills",
                    }
                ],
                "required_facts": ["Python"],
                "generation_method": "human",
                "review_status": "approved_gold",
                "tags": ["skills"],
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "salary-1",
                "question": "What salary is requested?",
                "answerable": False,
                "expected_evidence": [],
                "required_facts": [],
            }
        ),
        encoding="utf-8",
    )
    cases = load_cases(dataset)

    metrics = calculate_metrics(
        cases,
        [
            {
                "answer": "Python is listed [2].",
                "citations": [
                    {"source": "renamed.pdf", "page": 2, "text": "Other"},
                    {
                        "source": "renamed.pdf",
                        "source_sha256": source_sha256,
                        "page": 1,
                        "text": "Python",
                    },
                ],
            },
            {
                "answer": "I could not find enough relevant evidence.",
                "citations": [],
            },
        ],
        [0.1, 0.2],
    )

    assert metrics.hit_at_k == 1.0
    assert metrics.mean_reciprocal_rank == 0.5
    assert metrics.evidence_recall_at_k == 1.0
    assert metrics.required_evidence_coverage_at_k == 1.0
    assert metrics.ndcg_at_k == pytest.approx(1 / 1.584962500721156)
    assert metrics.citation_precision == 0.5
    assert metrics.fact_coverage == 1.0
    assert metrics.evidence_support == 1.0
    assert metrics.abstention_accuracy == 1.0
    assert metrics.false_answer_rate == 0.0
    assert metrics.false_abstention_rate == 0.0
    assert metrics.p50_latency_seconds == pytest.approx(0.15)
    assert metrics.p95_latency_seconds == 0.2


@pytest.mark.unit
@pytest.mark.parametrize(
    "answerable,required_facts",
    [("true", ["Python"]), (True, [])],
)
def test_rejects_ambiguous_golden_cases(
    tmp_path: Path, answerable: object, required_facts: list[str]
) -> None:
    dataset = tmp_path / "invalid.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "skills-1",
                "question": "What skills are listed?",
                "answerable": answerable,
                "expected_evidence": [{"source": "resume.pdf", "page": 1, "contains": ["Python"]}],
                "required_facts": required_facts,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises((TypeError, ValueError), match="answerable|required facts"):
        load_cases(dataset)


@pytest.mark.unit
def test_rejects_unreviewed_cases(tmp_path: Path) -> None:
    dataset = tmp_path / "silver.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "case_id": "draft-1",
                "question": "What is stored?",
                "answerable": True,
                "expected_evidence": [{"source": "notes.txt", "page": 1, "contains": ["locally"]}],
                "required_facts": ["locally"],
                "review_status": "synthetic_silver",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="approved_gold"):
        load_cases(dataset)


@pytest.mark.unit
def test_tracks_required_multi_passage_evidence() -> None:
    case = EvaluationCase(
        id="multi-1",
        question="Who approves access and when does it expire?",
        answerable=True,
        expected_evidence=(
            ExpectedEvidence("policy.pdf", 1, ("service owner",), evidence_group="approval"),
            ExpectedEvidence("policy.pdf", 2, ("24 hours",), evidence_group="expiry"),
        ),
        required_facts=("service owner", "24 hours"),
    )

    metrics = calculate_metrics(
        [case],
        [
            {
                "answer": "The service owner approves access.",
                "citations": [{"source": "policy.pdf", "page": 1, "text": "service owner"}],
            }
        ],
        [0.1],
    )

    assert metrics.evidence_recall_at_k == 0.5
    assert metrics.required_evidence_coverage_at_k == 0.5
    assert metrics.fact_coverage == 0.5
    assert metrics.evidence_support == 0.5
