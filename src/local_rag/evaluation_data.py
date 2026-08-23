"""Generate, validate, review, and release local RAG evaluation cases."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from local_rag.config import Settings
from local_rag.documents import load_document
from local_rag.domain import Page, RAGError
from local_rag.evaluation import load_cases
from local_rag.factory import build_completion_provider
from local_rag.providers import CompletionProvider

GENERATOR_SYSTEM_PROMPT = """Create one draft evaluation case from the supplied source page.
Use only the supplied evidence and required fact. Output exactly one JSON object with these fields:
question. The question must be naturally answered by the required fact, require the supplied
evidence, and not mention page numbers.
Do not wrap the JSON in Markdown."""

VALID_REVIEW_STATES = {"synthetic_silver", "approved_gold", "rejected", "needs_review"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


async def generate_candidates(
    source: Path,
    provider: CompletionProvider,
    *,
    count: int,
    dataset_version: str,
    split: str,
    corpus_id: str,
    corpus_version: str,
    generator_model: str,
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("count must be positive")
    pages = [page for page in load_document(source.name, source.read_bytes()) if page.text.strip()]
    if not pages:
        raise ValueError("source contains no usable pages")
    evidence_units = [
        (page, evidence_text) for page in pages for evidence_text in _evidence_passages(page.text)
    ]
    if not evidence_units:
        raise ValueError("source contains no stable prose evidence for direct-fact generation")

    candidates: list[dict[str, Any]] = []
    questions: set[str] = set()
    for index in range(count):
        page, evidence_text = evidence_units[index % len(evidence_units)]
        required_fact = _fact_from_evidence(evidence_text)
        if not required_fact:
            raise ValueError(f"could not select a stable fact from page {page.number}")
        last_error = ""
        for _attempt in range(3):
            response = await provider.complete(
                GENERATOR_SYSTEM_PROMPT,
                _generation_prompt(page, evidence_text, required_fact, questions, last_error),
                json_mode=True,
            )
            try:
                proposal = _parse_json_object(response)
                candidate = _build_candidate(
                    proposal,
                    page,
                    evidence_text,
                    required_fact,
                    dataset_version=dataset_version,
                    split=split,
                    corpus_id=corpus_id,
                    corpus_version=corpus_version,
                    generator_model=generator_model,
                )
                normalized_question = _normalize(str(candidate["question"]))
                if normalized_question in questions:
                    raise ValueError("generated question duplicates an earlier candidate")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                continue
            questions.add(normalized_question)
            candidates.append(candidate)
            break
        else:
            raise ValueError(
                f"could not generate a supported unique case for page {page.number}: {last_error}"
            )
    return candidates


def validate_candidate_records(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    for line_number, record in enumerate(records, 1):
        case_id = record.get("case_id", record.get("id"))
        question = record.get("question")
        prefix = f"case {case_id or line_number}"
        if not isinstance(case_id, str) or not case_id:
            errors.append(f"line {line_number}: missing case_id")
        elif case_id in seen_ids:
            errors.append(f"{prefix}: duplicate case_id")
        else:
            seen_ids.add(case_id)
        if not isinstance(question, str) or not question.strip():
            errors.append(f"{prefix}: missing question")
        else:
            normalized_question = _normalize(question)
            if normalized_question in seen_questions:
                errors.append(f"{prefix}: duplicate question")
            seen_questions.add(normalized_question)

        status = record.get("review_status")
        if status not in VALID_REVIEW_STATES:
            errors.append(f"{prefix}: invalid review_status")
        answerable = record.get("answerable")
        evidence = record.get("expected_evidence")
        facts = record.get("required_facts")
        answer = record.get("reference_answer")
        if not isinstance(answerable, bool):
            errors.append(f"{prefix}: answerable must be a boolean")
            continue
        if not answerable:
            if evidence or facts:
                errors.append(f"{prefix}: unanswerable cases must have empty evidence and facts")
            continue
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"{prefix}: answerable case requires expected_evidence")
            continue
        if (
            not isinstance(facts, list)
            or not facts
            or not all(isinstance(fact, str) for fact in facts)
        ):
            errors.append(f"{prefix}: answerable case requires string required_facts")
            continue
        if not isinstance(answer, str) or not answer:
            errors.append(f"{prefix}: answerable case requires reference_answer")
            continue

        evidence_text = " ".join(
            phrase
            for item in evidence
            if isinstance(item, dict)
            for phrase in item.get("contains", [])
            if isinstance(phrase, str)
        )
        if not evidence_text:
            errors.append(f"{prefix}: evidence requires stable text")
        if not all(_contains(evidence_text, fact) for fact in facts):
            errors.append(f"{prefix}: required facts are not all present in evidence")
        if not all(_contains(answer, fact) for fact in facts):
            errors.append(f"{prefix}: required facts are not all present in reference_answer")
        for item in evidence:
            if not isinstance(item, dict):
                errors.append(f"{prefix}: evidence entries must be objects")
                continue
            source_sha256 = item.get("source_sha256")
            if not isinstance(source_sha256, str) or not SHA256_RE.fullmatch(source_sha256):
                errors.append(f"{prefix}: evidence requires a valid source_sha256")
    return errors


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number} must contain a JSON object")
        records.append(cast(dict[str, Any], value))
    if not records:
        raise ValueError("dataset is empty")
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def review_candidate(
    path: Path,
    case_id: str,
    *,
    decision: str,
    reviewer: str,
    question: str | None = None,
    reference_answer: str | None = None,
    required_facts: list[str] | None = None,
) -> dict[str, Any]:
    records = read_jsonl(path)
    matches = [record for record in records if record.get("case_id", record.get("id")) == case_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one case named {case_id}")
    record = matches[0]
    if question is not None:
        record["question"] = question
    if reference_answer is not None:
        record["reference_answer"] = reference_answer
    if required_facts is not None:
        record["required_facts"] = required_facts
    record["reviewer"] = reviewer
    record["reviewed_at"] = _timestamp()
    if decision == "approve":
        record["review_status"] = "approved_gold"
        errors = validate_candidate_records([record])
        if errors:
            raise ValueError("; ".join(errors))
        record["approved_at"] = record["reviewed_at"]
    elif decision == "reject":
        record["review_status"] = "rejected"
    else:
        raise ValueError("decision must be approve or reject")
    write_jsonl(path, records, overwrite=True)
    return record


def release_dataset(source: Path, output: Path, *, overwrite: bool = False) -> int:
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; pass --force to replace it")
    approved = [
        record for record in read_jsonl(source) if record.get("review_status") == "approved_gold"
    ]
    if not approved:
        raise ValueError("dataset has no approved_gold cases")
    errors = validate_candidate_records(approved)
    if errors:
        raise ValueError("; ".join(errors))

    temporary = output.with_name(f".{output.name}.release.tmp")
    try:
        write_jsonl(temporary, approved, overwrite=True)
        load_cases(temporary)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return len(approved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="generate synthetic silver cases")
    generate.add_argument("source", type=Path)
    generate.add_argument("output", type=Path)
    generate.add_argument("--count", type=int, default=10)
    generate.add_argument("--dataset-version", default="0.1.0")
    generate.add_argument("--split", choices=["dev", "holdout", "smoke"], default="dev")
    generate.add_argument("--corpus-id", default="local")
    generate.add_argument("--corpus-version", default="1")
    generate.add_argument("--force", action="store_true")

    validate = commands.add_parser("validate", help="validate candidate records")
    validate.add_argument("dataset", type=Path)

    show = commands.add_parser("show", help="show a case for source-first review")
    show.add_argument("dataset", type=Path)
    show.add_argument("case_id")
    show.add_argument("--reveal-answer", action="store_true")

    review = commands.add_parser("review", help="approve, edit, or reject a candidate")
    review.add_argument("dataset", type=Path)
    review.add_argument("case_id")
    review.add_argument("--decision", choices=["approve", "reject"], required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--question")
    review.add_argument("--reference-answer")
    review.add_argument("--required-fact", action="append", dest="required_facts")

    release = commands.add_parser("release", help="freeze approved cases into a gold dataset")
    release.add_argument("dataset", type=Path)
    release.add_argument("output", type=Path)
    release.add_argument("--force", action="store_true")

    args = parser.parse_args()
    try:
        if args.command == "generate":
            settings = Settings.from_env()
            records = asyncio.run(
                generate_candidates(
                    args.source,
                    build_completion_provider(settings),
                    count=args.count,
                    dataset_version=args.dataset_version,
                    split=args.split,
                    corpus_id=args.corpus_id,
                    corpus_version=args.corpus_version,
                    generator_model=settings.chat_model,
                )
            )
            write_jsonl(args.output, records, overwrite=args.force)
            print(json.dumps({"generated": len(records), "output": str(args.output)}, indent=2))
        elif args.command == "validate":
            errors = validate_candidate_records(read_jsonl(args.dataset))
            print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
            raise SystemExit(1 if errors else 0)
        elif args.command == "show":
            record = _find_case(read_jsonl(args.dataset), args.case_id).copy()
            if not args.reveal_answer and "reference_answer" in record:
                record["reference_answer"] = "<hidden; pass --reveal-answer after source review>"
            print(json.dumps(record, indent=2, ensure_ascii=False))
        elif args.command == "review":
            record = review_candidate(
                args.dataset,
                args.case_id,
                decision=args.decision,
                reviewer=args.reviewer,
                question=args.question,
                reference_answer=args.reference_answer,
                required_facts=args.required_facts,
            )
            print(json.dumps(record, indent=2, ensure_ascii=False))
        elif args.command == "release":
            released = release_dataset(args.dataset, args.output, overwrite=args.force)
            print(json.dumps({"released": released, "output": str(args.output)}, indent=2))
    except (FileExistsError, OSError, RAGError, ValueError) as exc:
        parser.error(str(exc))


def _generation_prompt(
    page: Page,
    evidence_text: str,
    required_fact: str,
    questions: set[str],
    last_error: str,
) -> str:
    avoided = "\n".join(f"- {question}" for question in sorted(questions)) or "- none"
    retry = f"\nPrevious attempt failed because: {last_error}" if last_error else ""
    return (
        f"Source: {page.source}\nPage: {page.number}\n"
        f"Avoid duplicating these questions:\n{avoided}{retry}\n\n"
        f"Required fact (copying is not required):\n{required_fact}\n\n"
        f"Evidence passage:\n{evidence_text}"
    )


def _build_candidate(
    proposal: dict[str, Any],
    page: Page,
    evidence_text: str,
    required_fact: str,
    *,
    dataset_version: str,
    split: str,
    corpus_id: str,
    corpus_version: str,
    generator_model: str,
) -> dict[str, Any]:
    question = _required_string(proposal, "question")
    question_type = "direct_fact"
    if not page.source_sha256:
        raise ValueError("source page has no SHA-256 identity")

    normalized_evidence = _normalize(evidence_text)
    case_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{page.source_sha256}:{page.number}:{_normalize(question)}",
        )
    )
    return {
        "case_id": case_id,
        "dataset_version": dataset_version,
        "split": split,
        "corpus_id": corpus_id,
        "corpus_version": corpus_version,
        "question": question,
        "question_type": question_type,
        "answerable": True,
        "expected_evidence": [
            {
                "source": page.source,
                "source_sha256": page.source_sha256,
                "page": page.number,
                "contains": [evidence_text],
                "evidence_group": "A",
                "evidence_hash": hashlib.sha256(normalized_evidence.encode()).hexdigest(),
            }
        ],
        "required_facts": [required_fact],
        "reference_answer": required_fact,
        "document_ids": [page.source_sha256],
        "generation_method": "synthetic_local_llm",
        "generator_model": generator_model,
        "generator_prompt_version": "silver-v1",
        "automatic_validation": {
            "source_support_pass": True,
            "answer_support_pass": True,
            "closed_book_answerable": "not_checked",
        },
        "review_status": "synthetic_silver",
        "reviewer": None,
        "created_at": _timestamp(),
        "approved_at": None,
        "tags": [question_type, "synthetic"],
    }


def _parse_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model response did not contain a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise TypeError("model response must be a JSON object")
    return cast(dict[str, Any], value)


def _evidence_passages(text: str) -> list[str]:
    paragraphs = [
        paragraph.strip()[:1200]
        for paragraph in re.split(r"\n\s*\n", text)
        if len(_normalize(paragraph)) >= 120 and _fact_from_evidence(paragraph)
    ]
    if paragraphs:
        return paragraphs
    fallback = text.strip()[:1200]
    return [fallback] if fallback else []


def _fact_from_evidence(text: str) -> str:
    normalized = " ".join(text.replace("#", " ").split())
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    for sentence in sentences:
        has_verb = re.search(
            r"\b(?:is|are|was|were|be|been|has|have|had|can|could|will|would|"
            r"use|uses|used|show|shows|find|finds|found|introduce|introduces|"
            r"propose|proposes|achieve|achieves|outperform|outperforms)\b",
            sentence.casefold(),
        )
        if (
            20 <= len(sentence) <= 280
            and len(sentence.split()) >= 4
            and has_verb
            and "<" not in sentence
            and sentence.count("**") <= 2
            and sentence.count(",") <= 5
        ):
            return sentence
    if (
        20 <= len(normalized) <= 280
        and len(normalized.split()) >= 4
        and "<" not in normalized
        and normalized.count(",") <= 3
    ):
        return normalized
    return ""


def _find_case(records: list[dict[str, Any]], case_id: str) -> dict[str, Any]:
    matches = [record for record in records if record.get("case_id", record.get("id")) == case_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one case named {case_id}")
    return matches[0]


def _required_string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise TypeError(f"{key} must be a non-empty string")
    return item.strip()


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _contains(text: str, value: str) -> bool:
    return _normalize(value) in _normalize(text)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


if __name__ == "__main__":
    main()
