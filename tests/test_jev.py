import asyncio
import hashlib
import json
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from test_service import FakeChat, FakeStore, claim, service

from local_rag.api import create_app
from local_rag.config import Settings
from local_rag.domain import GroundedResponse, SearchResult
from local_rag.factory import build_service
from local_rag.jev_evaluation import load_audit_cases, replay_audits
from local_rag.providers import JevAuditor
from local_rag.service import RAGService


def response_data(
    ids: tuple[str, ...] = ("claim_1",),
    probabilities: dict[str, float] | None = None,
) -> dict[str, Any]:
    values = probabilities or {"supported": 0.97, "contradicted": 0.01, "insufficient": 0.02}
    return {
        "model": "jev-1.13.0",
        "answers": {
            name: {
                "type": "choice",
                "choice": max(values, key=lambda key: values[key]),
                "probabilities": values,
                "confidence": 0.91,
            }
            for name in ids
        },
        "usage": {"input_tokens": 500, "output_tokens": 30},
    }


@pytest.mark.unit
async def test_batches_only_identical_citation_sets_and_preserves_claim_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        assert request.headers["Authorization"] == "Bearer fake-key"
        payload = json.loads(request.content)
        requests.append(payload)
        assert "3" not in payload["state"]["cited_excerpts"]
        for name, question in payload["questions"].items():
            assert "claims." + name in question["instructions"]
        return httpx.Response(200, json=response_data(tuple(payload["questions"])))

    monkeypatch.setattr(
        httpx, "AsyncClient", partial(httpx.AsyncClient, transport=httpx.MockTransport(respond))
    )
    sources = {
        1: SearchResult("notes.txt", 1, "Rome is served.", 1),
        2: SearchResult("notes.txt", 2, "Milan is served.", 0.9),
        3: SearchResult("notes.txt", 3, "Uncited passage that could bias verification.", 0.8),
    }
    audits = await JevAuditor("fake-key").audit(
        [
            claim("Rome is served.", 1),
            claim("Milan is served.", 2),
            claim("Rome has service.", 1, 1),
            claim("Rome and Milan are served.", 1, 2),
        ],
        sources,
    )

    assert len(requests) == 3
    assert set(requests[0]["questions"]) == {"claim_1", "claim_3"}
    assert set(requests[0]["state"]["cited_excerpts"]) == {"1"}
    assert set(requests[1]["state"]["cited_excerpts"]) == {"2"}
    assert set(requests[2]["state"]["cited_excerpts"]) == {"1", "2"}
    assert [audit.claim_id for audit in audits] == [1, 2, 3, 4]
    assert audits[0].request_id == audits[2].request_id
    assert len({audit.request_id for audit in audits}) == 3
    assert all(audit.status == "completed" and audit.suggested_action == "keep" for audit in audits)
    assert audits[0].evidence_hashes == {1: hashlib.sha256(b"Rome is served.").hexdigest()}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("probabilities", "action"),
    [
        ({"supported": 0.90, "contradicted": 0.05, "insufficient": 0.05}, "keep"),
        ({"supported": 0.01, "contradicted": 0.98, "insufficient": 0.01}, "withhold"),
        ({"supported": 0.02, "contradicted": 0.01, "insufficient": 0.97}, "withhold"),
        ({"supported": 0.65, "contradicted": 0.10, "insufficient": 0.25}, "review"),
    ],
)
async def test_policy_uses_support_probability_not_confidence(
    monkeypatch: pytest.MonkeyPatch, probabilities: dict[str, float], action: str
) -> None:
    def respond(_request: httpx.Request) -> httpx.Response:
        data = response_data(probabilities=probabilities)
        data["answers"]["claim_1"]["confidence"] = 0.99
        return httpx.Response(200, json=data)

    monkeypatch.setattr(
        httpx, "AsyncClient", partial(httpx.AsyncClient, transport=httpx.MockTransport(respond))
    )
    audits = await JevAuditor("fake-key").audit(
        [claim("Assertion")], {1: SearchResult("notes.txt", 1, "Evidence", 1)}
    )
    assert audits[0].suggested_action == action


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("answers", "claim_1", "probabilities", "supported"), float("nan")),
        (("answers", "claim_1", "probabilities", "supported"), float("inf")),
        (("answers", "claim_1", "probabilities", "supported"), -0.1),
        (("answers", "claim_1", "probabilities", "supported"), 0.6),
        (("answers", "claim_1", "probabilities", "unexpected"), 0.0),
        (("answers", "claim_1", "confidence"), True),
        (("answers", "claim_1", "confidence"), "0.91"),
        (("answers", "claim_1", "type"), "noul"),
        (("answers", "claim_1", "choice"), "contradicted"),
        (("answers", "wrong_id"), {}),
        (("model",), "other-model"),
        (("usage", "input_tokens"), True),
    ],
)
async def test_invalid_provider_values_never_become_completed_audits(
    monkeypatch: pytest.MonkeyPatch, path: tuple[str, ...], value: object
) -> None:
    data = response_data()
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps(data).encode())

    monkeypatch.setattr(
        httpx, "AsyncClient", partial(httpx.AsyncClient, transport=httpx.MockTransport(respond))
    )
    audit = (
        await JevAuditor("fake-key").audit(
            [claim("Assertion")], {1: SearchResult("notes.txt", 1, "Evidence", 1)}
        )
    )[0]
    assert audit.status == "unavailable"
    assert audit.suggested_action == "unavailable"
    assert audit.assessment is None
    assert audit.model is None


@pytest.mark.unit
@pytest.mark.parametrize("status", [401, 429, 529])
async def test_remote_error_bodies_and_credentials_are_not_exposed(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="private source excerpt and fake-secret")

    monkeypatch.setattr(
        httpx, "AsyncClient", partial(httpx.AsyncClient, transport=httpx.MockTransport(respond))
    )
    audit = (
        await JevAuditor("fake-secret").audit(
            [claim("Assertion")], {1: SearchResult("notes.txt", 1, "Evidence", 1)}
        )
    )[0]
    assert audit.status == "unavailable"
    assert f"HTTP {status}" in (audit.error or "")
    assert "fake-secret" not in audit.model_dump_json()
    assert "private source excerpt" not in audit.model_dump_json()


@pytest.mark.unit
async def test_total_budget_preserves_completed_results_and_marks_remaining_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 2:
            await asyncio.Event().wait()
        payload = json.loads(request.content)
        return httpx.Response(200, json=response_data(tuple(payload["questions"])))

    monkeypatch.setattr(
        httpx, "AsyncClient", partial(httpx.AsyncClient, transport=httpx.MockTransport(respond))
    )
    audits = await asyncio.wait_for(
        JevAuditor("fake-key", timeout=0.1).audit(
            [claim("First", 1), claim("Second", 2), claim("Third", 3)],
            {number: SearchResult("notes.txt", number, "Evidence", 1) for number in (1, 2, 3)},
        ),
        timeout=2,
    )
    assert calls == 2
    assert [audit.status for audit in audits] == ["completed", "unavailable", "unavailable"]
    assert all(audit.assessment is None for audit in audits[1:])


@pytest.mark.unit
async def test_missing_citations_and_empty_claims_never_send_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def respond(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("No request may be sent")

    monkeypatch.setattr(
        httpx, "AsyncClient", partial(httpx.AsyncClient, transport=httpx.MockTransport(respond))
    )
    auditor = JevAuditor("fake-key")
    assert await auditor.audit([], {}) == []
    audit = (await auditor.audit([claim("Assertion", 2)], {}))[0]
    assert audit.status == "unavailable"
    assert audit.error == "A claim references unavailable citation IDs."


@pytest.mark.unit
async def test_observer_keeps_answer_and_ledger_while_using_global_citation_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries = ["Explain silver candidates", "describe human review"]
    silver = SearchResult("notes.txt", 1, "Silver candidates are draft questions.", 1)
    review = SearchResult("notes.txt", 2, "Human review approves or rejects candidates.", 0.9)
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json=response_data(
                tuple(payload["questions"]),
                {"supported": 0.02, "contradicted": 0.01, "insufficient": 0.97},
            ),
        )

    monkeypatch.setattr(
        httpx, "AsyncClient", partial(httpx.AsyncClient, transport=httpx.MockTransport(respond))
    )

    def make_service() -> RAGService:
        return service(
            FakeStore(matches_by_query={queries[0]: [silver], queries[1]: [review]}),
            FakeChat(
                [
                    GroundedResponse(claims=[claim("Silver candidates are draft questions.")]),
                    GroundedResponse(
                        claims=[claim("Human review approves or rejects candidates.")]
                    ),
                ]
            ),
        )

    baseline = await make_service().ask("Explain silver candidates and describe human review")
    observed_service = make_service()
    observed_service.auditor = JevAuditor("fake-key")
    observed = await observed_service.ask("Explain silver candidates and describe human review")
    assert observed.text == baseline.text
    assert observed.citations == baseline.citations
    assert [audit.claim_id for audit in observed.audits] == [1, 2]
    assert [audit.citations for audit in observed.audits] == [[1], [2]]
    assert all(audit.suggested_action == "withhold" for audit in observed.audits)
    assert set(requests[1]["state"]["cited_excerpts"]) == {"2"}
    assert requests[1]["state"]["cited_excerpts"]["2"] == review.text


@pytest.mark.unit
async def test_repair_rechecks_individual_passages_and_changes_only_the_citation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    passages = [
        SearchResult("aes.pdf", 19, "The block size of AES-256 is discussed in Table 3.", 1),
        SearchResult("aes.pdf", 20, "This section discusses AES transformations.", 0.9),
        SearchResult("aes.pdf", 13, "An AES block is 16 bytes.", 0.8),
    ]
    requested_ids: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        excerpts = payload["state"]["cited_excerpts"]
        assert len(excerpts) == 1
        citation = next(iter(excerpts))
        requested_ids.append(citation)
        values = (
            {"supported": 0.99, "contradicted": 0.0, "insufficient": 0.01}
            if citation == "3"
            else {"supported": 0.15, "contradicted": 0.0, "insufficient": 0.85}
        )
        return httpx.Response(200, json=response_data(tuple(payload["questions"]), values))

    monkeypatch.setattr(
        httpx, "AsyncClient", partial(httpx.AsyncClient, transport=httpx.MockTransport(respond))
    )
    rag = service(
        FakeStore(passages),
        FakeChat([GroundedResponse(claims=[claim("The block size of AES-256 is 16 bytes.")])]),
    )
    rag.auditor = JevAuditor("fake-key")
    rag.repair_citations = True
    answer = await rag.ask("What is the block size of AES-256?")

    assert answer.text == "The block size of AES-256 is 16 bytes. [3]"
    assert answer.citations == passages
    audit = answer.audits[0]
    assert audit.claim_id == 1 and audit.citations == [3]
    assert audit.suggested_action == "keep"
    assert audit.original is not None and audit.original.citations == [1]
    assert audit.original.assessment is not None
    assert audit.original.assessment.probabilities.insufficient == 0.85
    assert [attempt.citations for attempt in audit.repair_attempts] == [[2], [3]]
    assert audit.repair_attempts[-1].evidence_hashes == {
        3: hashlib.sha256(b"An AES block is 16 bytes.").hexdigest()
    }
    assert requested_ids == ["1", "2", "3"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "outcome", ["supported", "contradicted", "initial_unavailable", "weak", "candidate_unavailable"]
)
def test_repair_api_retains_original_citations_when_no_replacement_is_verified(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = json.loads(request.content)
        if (calls == 1 and outcome == "initial_unavailable") or (
            calls == 2 and outcome == "candidate_unavailable"
        ):
            return httpx.Response(529, text="private remote error")
        if calls == 1:
            values = {
                "supported": {"supported": 0.99, "contradicted": 0.0, "insufficient": 0.01},
                "contradicted": {"supported": 0.01, "contradicted": 0.99, "insufficient": 0.0},
            }.get(outcome, {"supported": 0.15, "contradicted": 0.0, "insufficient": 0.85})
        else:
            values = {"supported": 0.89, "contradicted": 0.0, "insufficient": 0.11}
        return httpx.Response(200, json=response_data(tuple(payload["questions"]), values))

    monkeypatch.setattr(
        httpx, "AsyncClient", partial(httpx.AsyncClient, transport=httpx.MockTransport(respond))
    )
    rag = service(
        FakeStore(
            [
                SearchResult("aes.pdf", 19, "The block size of AES-256 is discussed elsewhere.", 1),
                SearchResult("aes.pdf", 20, "A related AES section.", 0.9),
            ]
        ),
        FakeChat([GroundedResponse(claims=[claim("The block size of AES-256 is 16 bytes.")])]),
    )
    rag.auditor = JevAuditor("fake-key")
    rag.repair_citations = True
    settings = replace(Settings.from_env(), jev_mode="repair", jev_api_key="fake-key")
    with TestClient(create_app(settings, rag)) as client:
        response = client.post(
            "/api/query", json={"question": "What is the block size of AES-256?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["audit_mode"] == "repair"
        assert data["answer"] == "The block size of AES-256 is 16 bytes. [1]"
        assert data["audits"][0]["citations"] == [1]
        assert "private remote error" not in response.text
        assert client.get("/health").json()["jev_mode"] == "repair"
        if outcome in {"supported", "contradicted", "initial_unavailable"}:
            assert calls == 1
            assert data["audits"][0]["original"] is None
        else:
            assert calls == 2
            assert data["audits"][0]["original"]["citations"] == [1]
            assert len(data["audits"][0]["repair_attempts"]) == 1


@pytest.mark.unit
async def test_multipart_repair_preserves_global_ids_and_claim_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        values = (
            {"supported": 0.15, "contradicted": 0.0, "insufficient": 0.85}
            if "2" in payload["state"]["cited_excerpts"]
            else {"supported": 0.99, "contradicted": 0.0, "insufficient": 0.01}
        )
        return httpx.Response(200, json=response_data(tuple(payload["questions"]), values))

    monkeypatch.setattr(
        httpx, "AsyncClient", partial(httpx.AsyncClient, transport=httpx.MockTransport(respond))
    )
    block = SearchResult("aes.pdf", 19, "The block size is 16 bytes.", 1)
    vague = SearchResult("policy.txt", 1, "The cancellation window is discussed elsewhere.", 1)
    window = SearchResult("policy.txt", 2, "Customers can cancel within thirty days.", 0.9)
    rag = service(
        FakeStore(
            matches_by_query={
                "Explain block size": [block],
                "describe cancellation window": [vague, window],
            }
        ),
        FakeChat(
            [
                GroundedResponse(claims=[claim("The block size is 16 bytes.")]),
                GroundedResponse(claims=[claim("Customers can cancel within thirty days.")]),
            ]
        ),
    )
    rag.auditor = JevAuditor("fake-key")
    rag.repair_citations = True
    answer = await rag.ask("Explain block size and describe cancellation window")
    assert "The block size is 16 bytes. [1]" in answer.text
    assert "Customers can cancel within thirty days. [3]" in answer.text
    assert [audit.claim_id for audit in answer.audits] == [1, 2]
    assert answer.audits[1].original is not None
    assert answer.audits[1].original.citations == [2]
    assert set(requests[-1]["questions"]) == {"claim_2"}
    assert requests[-1]["state"]["cited_excerpts"] == {"3": window.text}


@pytest.mark.unit
async def test_repair_does_not_borrow_evidence_from_another_question_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        excerpts = payload["state"]["cited_excerpts"]
        submitted.extend(excerpts.values())
        assert "An AES block is 16 bytes." not in excerpts.values()
        return httpx.Response(
            200,
            json=response_data(
                tuple(payload["questions"]),
                {"supported": 0.15, "contradicted": 0.0, "insufficient": 0.85},
            ),
        )

    monkeypatch.setattr(
        httpx, "AsyncClient", partial(httpx.AsyncClient, transport=httpx.MockTransport(respond))
    )
    rag = service(
        FakeStore(
            matches_by_query={
                "Explain block size": [
                    SearchResult("aes.pdf", 19, "The block size is discussed elsewhere.", 1),
                    SearchResult("aes.pdf", 20, "A related section.", 0.9),
                ],
                "describe cancellation window": [
                    SearchResult("other.txt", 1, "An AES block is 16 bytes.", 1)
                ],
            }
        ),
        FakeChat(
            [
                GroundedResponse(claims=[claim("The block size is 16 bytes.")]),
                GroundedResponse(unsupported=["No cancellation window is stated."]),
            ]
        ),
    )
    rag.auditor = JevAuditor("fake-key")
    rag.repair_citations = True
    answer = await rag.ask("Explain block size and describe cancellation window")
    assert "The block size is 16 bytes. [1]" in answer.text
    assert len(submitted) == 2
    assert answer.audits[0].suggested_action == "review"


@pytest.mark.unit
async def test_repair_shares_the_initial_audit_time_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(0.25)
        else:
            await asyncio.Event().wait()
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json=response_data(
                tuple(payload["questions"]),
                {"supported": 0.15, "contradicted": 0.0, "insufficient": 0.85},
            ),
        )

    monkeypatch.setattr(
        httpx, "AsyncClient", partial(httpx.AsyncClient, transport=httpx.MockTransport(respond))
    )
    rag = service(
        FakeStore(
            [
                SearchResult("aes.pdf", 19, "The block size is discussed elsewhere.", 1),
                SearchResult("aes.pdf", 20, "First candidate.", 0.9),
                SearchResult("aes.pdf", 21, "Second candidate.", 0.8),
            ]
        ),
        FakeChat([GroundedResponse(claims=[claim("The block size is 16 bytes.")])]),
    )
    rag.auditor = JevAuditor("fake-key", timeout=0.4)
    rag.repair_citations = True
    answer = await asyncio.wait_for(rag.ask("Explain block size"), timeout=0.55)
    assert calls == 2
    assert answer.text == "The block size is 16 bytes. [1]"
    assert answer.audits[0].status == "completed"
    assert answer.audits[0].repair_attempts[0].status == "unavailable"


@pytest.mark.unit
def test_api_returns_audit_metadata_and_degrades_without_changing_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def respond(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("fake-secret and private provider details")

    monkeypatch.setattr(
        httpx, "AsyncClient", partial(httpx.AsyncClient, transport=httpx.MockTransport(respond))
    )
    rag = service(FakeStore([SearchResult("notes.txt", 1, "Evidence", 1)]))
    rag.auditor = JevAuditor("fake-secret")
    settings = replace(Settings.from_env(), jev_mode="observe", jev_api_key="fake-secret")
    with TestClient(create_app(settings, rag)) as client:
        response = client.post("/api/query", json={"question": "Where is evidence?"})
        assert response.status_code == 200
        data = response.json()
        assert data["audit_mode"] == "observe"
        assert data["answer"] == "Grounded answer for Where is evidence? [1]"
        assert data["audits"][0]["status"] == "unavailable"
        assert data["audits"][0]["assessment"] is None
        assert "fake-secret" not in response.text
        assert client.get("/health").json()["jev_mode"] == "observe"


@pytest.mark.unit
def test_observer_configuration_is_opt_in_and_does_not_expose_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("RAG_JEV_MODE", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-secret")
    settings = replace(Settings.from_env(), data_dir=tmp_path / "qdrant")
    assert settings.jev_mode == "off"
    assert "fake-secret" not in repr(settings)
    rag = build_service(settings)
    assert rag.auditor is None
    rag.close()
    monkeypatch.setenv("RAG_JEV_MODE", "observe")
    settings = replace(Settings.from_env(), data_dir=tmp_path / "qdrant")
    rag = build_service(settings)
    assert isinstance(rag.auditor, JevAuditor)
    assert not rag.repair_citations
    rag.close()
    monkeypatch.setenv("RAG_JEV_MODE", "repair")
    settings = replace(Settings.from_env(), data_dir=tmp_path / "qdrant")
    rag = build_service(settings)
    assert isinstance(rag.auditor, JevAuditor)
    assert rag.repair_citations
    rag.close()
    monkeypatch.delenv("TYPESAFE_API_KEY")
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        Settings.from_env()


@pytest.mark.unit
async def test_smoke_replay_reports_missed_labels_and_provider_unavailability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = load_audit_cases(Path("evaluation/jev-smoke.jsonl"))[:2]
    calls = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return (
            httpx.Response(200, json=response_data())
            if calls == 1
            else httpx.Response(529, text="provider overloaded")
        )

    monkeypatch.setattr(
        httpx, "AsyncClient", partial(httpx.AsyncClient, transport=httpx.MockTransport(respond))
    )
    report = await replay_audits(cases, JevAuditor("fake-key"))
    assert report["summary"]["cases"] == 2
    assert report["summary"]["completed"] == 1
    assert report["summary"]["correct"] == 1
    assert report["summary"]["label_accuracy"] == 0.5
    assert report["summary"]["input_tokens"] == 500
    assert report["results"][1]["audit"]["status"] == "unavailable"


@pytest.mark.unit
def test_smoke_dataset_rejects_duplicate_case_ids_and_missing_citations(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    first = Path("evaluation/jev-smoke.jsonl").read_text(encoding="utf-8").splitlines()[0]
    path.write_text(first + "\n" + first, encoding="utf-8")
    with pytest.raises(ValueError, match="unique case IDs"):
        load_audit_cases(path)
    data = json.loads(first)
    data["claim"]["citations"] = [2]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="include every cited ID"):
        load_audit_cases(path)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"jev_mode": "enforce"}, "Jev mode"),
        ({"jev_model": ""}, "Jev model"),
        ({"jev_timeout": float("inf")}, "Jev timeout"),
        ({"jev_threshold": float("nan")}, "Jev threshold"),
        ({"jev_threshold": 0.5}, "Jev threshold"),
    ],
)
def test_invalid_jev_settings_cannot_disable_the_boundaries(
    updates: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(Settings.from_env(), **updates).validate()
