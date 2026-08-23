from dataclasses import dataclass
from typing import Any

import pytest

from local_rag.evaluation import EvaluationCase
from local_rag.ragas_evaluation import Scorers, score_responses


@dataclass
class Result:
    value: float


class FakeMetric:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[dict[str, Any]] = []

    async def ascore(self, **kwargs: Any) -> Result:
        self.calls.append(kwargs)
        return Result(self.value)


@pytest.mark.unit
async def test_scores_answerable_responses_with_local_judges() -> None:
    faithfulness = FakeMetric(0.9)
    precision = FakeMetric(0.8)
    recall = FakeMetric(0.7)
    correctness = FakeMetric(0.6)
    case = EvaluationCase(
        id="case-1",
        question="Where is the evidence?",
        answerable=True,
        expected_evidence=(),
        required_facts=("local",),
        reference_answer="The evidence is local.",
    )

    metrics = await score_responses(
        [case],
        [{"answer": "It is local.", "citations": [{"text": "Evidence is local."}]}],
        Scorers(faithfulness, precision, recall, correctness),
    )

    assert metrics.faithfulness == 0.9
    assert metrics.context_precision == 0.8
    assert metrics.context_recall == 0.7
    assert metrics.factual_correctness == 0.6
    assert faithfulness.calls[0]["retrieved_contexts"] == ["Evidence is local."]
