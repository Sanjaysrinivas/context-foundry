import hashlib
import json
import sys
from pathlib import Path

import pytest

from local_rag.evaluation import load_cases
from local_rag.evaluation_data import (
    generate_candidates,
    main,
    release_dataset,
    review_candidate,
    validate_candidate_records,
    write_jsonl,
)


class FakeGenerator:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = responses
        self.calls = 0

    async def complete(
        self, system_prompt: str, user_prompt: str, *, json_mode: bool = False
    ) -> str:
        assert "required fact" in system_prompt
        assert "Evidence passage:" in user_prompt
        assert json_mode
        response = self.responses[self.calls]
        self.calls += 1
        return json.dumps(response)


def proposal() -> dict[str, object]:
    return {
        "question": "Where is the evidence stored?",
    }


@pytest.mark.unit
async def test_generates_valid_silver_candidate_and_retries(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    content = b"The evidence is stored locally on this computer."
    source.write_bytes(content)
    provider = FakeGenerator([{}, proposal()])

    records = await generate_candidates(
        source,
        provider,
        count=1,
        dataset_version="0.1.0",
        split="dev",
        corpus_id="notes",
        corpus_version="1",
        generator_model="fake-local",
    )

    assert provider.calls == 2
    assert validate_candidate_records(records) == []
    assert records[0]["review_status"] == "synthetic_silver"
    assert records[0]["generation_method"] == "synthetic_local_llm"
    expected_hash = hashlib.sha256(content).hexdigest()
    assert records[0]["document_ids"] == [expected_hash]
    assert records[0]["expected_evidence"][0]["source_sha256"] == expected_hash


@pytest.mark.unit
async def test_reviews_and_releases_approved_gold(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("The evidence is stored locally.", encoding="utf-8")
    records = await generate_candidates(
        source,
        FakeGenerator([proposal()]),
        count=1,
        dataset_version="0.1.0",
        split="holdout",
        corpus_id="notes",
        corpus_version="1",
        generator_model="fake-local",
    )
    candidates = tmp_path / "candidates.jsonl"
    gold = tmp_path / "gold.jsonl"
    write_jsonl(candidates, records)

    reviewed = review_candidate(
        candidates,
        str(records[0]["case_id"]),
        decision="approve",
        reviewer="sanjay",
    )
    released = release_dataset(candidates, gold)

    assert reviewed["review_status"] == "approved_gold"
    assert reviewed["reviewer"] == "sanjay"
    assert reviewed["approved_at"]
    assert released == 1
    assert load_cases(gold)[0].review_status == "approved_gold"


@pytest.mark.unit
async def test_generation_skips_markup_heavy_front_matter(tmp_path: Path) -> None:
    source = tmp_path / "paper.txt"
    source.write_text(
        "**Author One**<sup>1</sup>, **Author Two**<sup>2</sup>, **Author Three**<sup>3</sup>\n\n"
        "Large language models have been shown to store factual knowledge in their parameters. "
        "This behavior motivates evaluating whether external evidence changes their answers.",
        encoding="utf-8",
    )

    records = await generate_candidates(
        source,
        FakeGenerator([proposal()]),
        count=1,
        dataset_version="0.1.0",
        split="dev",
        corpus_id="paper",
        corpus_version="1",
        generator_model="fake-local",
    )

    assert "Large language models" in records[0]["required_facts"][0]


@pytest.mark.unit
def test_validation_reports_duplicate_and_unsupported_cases() -> None:
    candidate = {
        "case_id": "case-1",
        "question": "Where?",
        "answerable": True,
        "expected_evidence": [
            {
                "source": "notes.txt",
                "source_sha256": "a" * 64,
                "page": 1,
                "contains": ["stored remotely"],
            }
        ],
        "required_facts": ["stored locally"],
        "reference_answer": "It is stored locally.",
        "review_status": "synthetic_silver",
    }

    errors = validate_candidate_records([candidate, candidate.copy()])

    assert any("duplicate case_id" in error for error in errors)
    assert any("duplicate question" in error for error in errors)
    assert any("not all present in evidence" in error for error in errors)


@pytest.mark.unit
async def test_cli_source_first_review_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("The evidence is stored locally.", encoding="utf-8")
    records = await generate_candidates(
        source,
        FakeGenerator([proposal()]),
        count=1,
        dataset_version="0.1.0",
        split="holdout",
        corpus_id="notes",
        corpus_version="1",
        generator_model="fake-local",
    )
    candidates = tmp_path / "candidates.jsonl"
    gold = tmp_path / "gold.jsonl"
    write_jsonl(candidates, records)
    case_id = str(records[0]["case_id"])

    monkeypatch.setattr(sys, "argv", ["local-rag-eval-data", "validate", str(candidates)])
    with pytest.raises(SystemExit) as validation:
        main()
    assert validation.value.code == 0
    assert '"valid": true' in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        ["local-rag-eval-data", "show", str(candidates), case_id],
    )
    main()
    assert "<hidden; pass --reveal-answer" in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "local-rag-eval-data",
            "review",
            str(candidates),
            case_id,
            "--decision",
            "approve",
            "--reviewer",
            "sanjay",
        ],
    )
    main()
    assert '"review_status": "approved_gold"' in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        ["local-rag-eval-data", "release", str(candidates), str(gold)],
    )
    main()
    assert '"released": 1' in capsys.readouterr().out
    assert load_cases(gold)[0].id == case_id
