import json
from pathlib import Path

import pytest

from local_rag.evaluation import calculate_metrics, load_cases


@pytest.mark.unit
def test_loads_jsonl_and_calculates_quality_metrics(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "skills-1",
                "question": "What skills are listed?",
                "answerable": True,
                "expected_evidence": [{"source": "resume.pdf", "page": 1, "contains": ["Python"]}],
                "required_facts": ["Python"],
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
                    {"source": "resume.pdf", "page": 2, "text": "Other"},
                    {"source": "resume.pdf", "page": 1, "text": "Python"},
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
    assert metrics.citation_precision == 0.5
    assert metrics.fact_coverage == 1.0
    assert metrics.evidence_support == 1.0
    assert metrics.abstention_accuracy == 1.0
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
