"""Replay explicit claim/citation labels against Jev without an answering model."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from local_rag.config import Settings
from local_rag.domain import AuditRelation, GroundedClaim, SearchResult
from local_rag.providers import JevAuditor


class AuditPassage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: int = Field(ge=1)
    text: str = Field(min_length=1)


class AuditCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    case_id: str = Field(min_length=1)
    claim: GroundedClaim
    passages: list[AuditPassage] = Field(min_length=1)
    expected_relation: AuditRelation

    @model_validator(mode="after")
    def require_cited_passages(self) -> AuditCase:
        ids = [passage.id for passage in self.passages]
        if len(set(ids)) != len(ids) or not set(self.claim.citations) <= set(ids):
            raise ValueError("passage IDs must be unique and include every cited ID")
        return self


def load_audit_cases(path: Path) -> list[AuditCase]:
    cases = [
        AuditCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("audit dataset must be non-empty with unique case IDs")
    return cases


async def replay_audits(cases: list[AuditCase], auditor: JevAuditor) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for case in cases:
        sources = {
            passage.id: SearchResult("audit-case", passage.id, passage.text, 0.0)
            for passage in case.passages
        }
        audit = (await auditor.audit([case.claim], sources))[0]
        records.append(
            {
                "case_id": case.case_id,
                "expected_relation": case.expected_relation,
                "correct": audit.assessment is not None
                and audit.assessment.choice == case.expected_relation,
                "audit": audit.model_dump(),
            }
        )
    return {
        "summary": {
            "cases": len(records),
            "completed": sum(record["audit"]["status"] == "completed" for record in records),
            "correct": sum(record["correct"] for record in records),
            "label_accuracy": (
                sum(record["correct"] for record in records) / len(records) if records else None
            ),
            "unsupported_accepted": sum(
                record["expected_relation"] != "supported"
                and record["audit"]["suggested_action"] == "keep"
                for record in records
            ),
            "supported_withheld": sum(
                record["expected_relation"] == "supported"
                and record["audit"]["suggested_action"] == "withhold"
                for record in records
            ),
            "needs_review": sum(
                record["audit"]["suggested_action"] == "review" for record in records
            ),
            "input_tokens": sum(record["audit"]["input_tokens"] or 0 for record in records),
            "elapsed_seconds": time.perf_counter() - started,
            "threshold": auditor.threshold,
            "requested_model": auditor.model,
        },
        "results": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path, default=Path("evaluation/jev-smoke.jsonl"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        settings = Settings.from_env()
        if not settings.jev_api_key.strip():
            raise ValueError("Set TYPESAFE_API_KEY locally before running live Jev evaluation")
        cases = load_audit_cases(args.dataset)
        report = asyncio.run(
            replay_audits(
                cases,
                JevAuditor(
                    settings.jev_api_key,
                    settings.jev_model,
                    settings.jev_timeout,
                    settings.jev_threshold,
                ),
            )
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report["summary"], indent=2))
    for record in report["results"]:
        audit = record["audit"]
        relation = audit["assessment"]["choice"] if audit["assessment"] else "unavailable"
        print(f"{record['case_id']}: expected={record['expected_relation']}, observed={relation}")
    raise SystemExit(0 if report["summary"]["correct"] == len(cases) else 1)


if __name__ == "__main__":
    main()
