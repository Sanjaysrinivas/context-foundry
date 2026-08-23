"""Run optional local Ragas judges over the human-approved evaluation dataset."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from local_rag.domain import RAGError
from local_rag.evaluation import EvaluationCase, load_cases, query_cases


class Metric(Protocol):
    async def ascore(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class Scorers:
    faithfulness: Metric
    context_precision: Metric
    context_recall: Metric
    factual_correctness: Metric


@dataclass(frozen=True, slots=True)
class RagasMetrics:
    cases: int
    faithfulness: float
    context_precision: float
    context_recall: float
    factual_correctness: float


async def score_responses(
    cases: list[EvaluationCase],
    responses: list[dict[str, Any]],
    scorers: Scorers,
) -> RagasMetrics:
    if len(cases) != len(responses):
        raise ValueError("Cases and responses must have equal lengths")
    scores: dict[str, list[float]] = {
        "faithfulness": [],
        "context_precision": [],
        "context_recall": [],
        "factual_correctness": [],
    }
    for case, response in zip(cases, responses, strict=True):
        if not case.answerable:
            continue
        answer = str(response.get("answer", ""))
        contexts = [str(item.get("text", "")) for item in response.get("citations", [])]
        reference = case.reference_answer or " ".join(case.required_facts)
        calls = {
            "faithfulness": scorers.faithfulness.ascore(
                user_input=case.question,
                response=answer,
                retrieved_contexts=contexts,
            ),
            "context_precision": scorers.context_precision.ascore(
                user_input=case.question,
                reference=reference,
                retrieved_contexts=contexts,
            ),
            "context_recall": scorers.context_recall.ascore(
                user_input=case.question,
                reference=reference,
                retrieved_contexts=contexts,
            ),
            "factual_correctness": scorers.factual_correctness.ascore(
                response=answer,
                reference=reference,
            ),
        }
        for name, result in zip(calls, await asyncio.gather(*calls.values()), strict=True):
            value = getattr(result, "value", None)
            if not isinstance(value, int | float):
                raise RAGError(f"Ragas returned no numeric {name} score")
            scores[name].append(float(value))

    evaluated = len(scores["faithfulness"])
    if not evaluated:
        raise ValueError("Ragas evaluation requires at least one answerable case")
    return RagasMetrics(
        cases=evaluated,
        faithfulness=statistics.fmean(scores["faithfulness"]),
        context_precision=statistics.fmean(scores["context_precision"]),
        context_recall=statistics.fmean(scores["context_recall"]),
        factual_correctness=statistics.fmean(scores["factual_correctness"]),
    )


def build_scorers(base_url: str, model: str) -> tuple[Scorers, Any]:
    try:
        openai = importlib.import_module("openai")
        ragas_llms = importlib.import_module("ragas.llms")
        metrics = importlib.import_module("ragas.metrics.collections")
    except ImportError as exc:
        raise RAGError(
            "Ragas evaluation is optional; run with `uv run --extra evaluation "
            "local-rag-eval-ragas ...`"
        ) from exc

    client = openai.AsyncOpenAI(api_key="ollama", base_url=base_url.rstrip("/"))
    llm = ragas_llms.llm_factory(model, provider="openai", client=client)
    return (
        Scorers(
            faithfulness=metrics.Faithfulness(llm=llm),
            context_precision=metrics.ContextPrecision(llm=llm),
            context_recall=metrics.ContextRecall(llm=llm),
            factual_correctness=metrics.FactualCorrectness(llm=llm),
        ),
        client,
    )


async def run(args: argparse.Namespace) -> RagasMetrics:
    cases = load_cases(args.dataset)
    responses, _latencies = await query_cases(cases, args.base_url, args.timeout)
    scorers, client = build_scorers(args.judge_base_url, args.judge_model)
    try:
        try:
            return await score_responses(cases, responses, scorers)
        except (RAGError, ValueError):
            raise
        except Exception as exc:
            raise RAGError(
                "Ragas judge failed to produce its required schema. "
                "Use a stronger local judge such as qwen3:8b."
            ) from exc
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--judge-base-url",
        default=f"{os.getenv('RAG_OLLAMA_BASE_URL', 'http://localhost:11434').rstrip('/')}/v1",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("RAG_EVAL_MODEL", "qwen3:8b"),
    )
    args = parser.parse_args()
    try:
        metrics = asyncio.run(run(args))
    except (OSError, RAGError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(asdict(metrics), indent=2))


if __name__ == "__main__":
    main()
