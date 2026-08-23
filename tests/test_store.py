from pathlib import Path

import pytest

from local_rag.domain import Chunk
from local_rag.store import QdrantVectorStore, _reciprocal_rank_fusion, lexical_tokens


def test_lexical_tokens_match_simple_plural_variants() -> None:
    assert lexical_tokens("stable evidence anchors and categories") == lexical_tokens(
        "stable evidence anchor and category"
    )


def test_reciprocal_rank_fusion_rewards_results_found_by_both_retrievers() -> None:
    ranked = _reciprocal_rank_fusion(
        ["dense-only", "shared"],
        ["shared", "lexical-only"],
    )

    assert ranked[0][0] == "shared"
    assert 0 < ranked[0][1] <= 1


@pytest.mark.integration
def test_local_qdrant_round_trip(tmp_path: Path) -> None:
    store = QdrantVectorStore(tmp_path / "qdrant", "test-documents")
    chunk = Chunk(
        "292c8e7d-3fc3-43c7-8034-2026408ed2af",
        "doc-1",
        "notes.txt",
        1,
        0,
        "Local evidence",
        "a" * 64,
    )

    store.replace([chunk], [[1.0, 0.0]])
    results = store.search([1.0, 0.0], "local evidence", limit=1, threshold=0.1)

    assert len(results) == 1
    assert results[0].text == "Local evidence"
    assert results[0].score == pytest.approx(1.0)
    assert results[0].source_sha256 == "a" * 64
    assert store.list_documents()[0].document_id == "doc-1"
    assert store.list_documents()[0].source_sha256 == "a" * 64
    lexical_fallback = store.search([0.0, 1.0], "local evidence", limit=1, threshold=0.9)
    assert lexical_fallback[0].text == "Local evidence"

    same_document = Chunk(
        "492457a1-309a-46c9-b926-e7a1f94ce4bc",
        "doc-1",
        "notes.txt",
        1,
        0,
        "Reindexed evidence",
        "a" * 64,
    )
    store.replace([same_document], [[1.0, 0.0]])
    assert store.list_documents()[0].chunks == 1
    assert store.search([1.0, 0.0], "reindexed", limit=1, threshold=0.1)[0].text == (
        "Reindexed evidence"
    )

    replacement = Chunk(
        "49f4b5f6-b709-49cd-a460-f43a5f216e05",
        "doc-2",
        "notes.txt",
        1,
        0,
        "Replacement evidence",
    )
    store.replace([replacement], [[1.0, 0.0]])
    assert [document.document_id for document in store.list_documents()] == ["doc-2"]
    assert (
        store.search([1.0, 0.0], "replacement", limit=1, threshold=0.1, document_ids=["doc-1"])
        == []
    )
    assert store.delete_document("doc-2")
    assert not store.delete_document("doc-2")

    store.clear()
    assert store.search([1.0, 0.0], "local", limit=1, threshold=0.1) == []
    store.close()
