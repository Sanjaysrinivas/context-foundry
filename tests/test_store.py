from pathlib import Path

import pytest

from local_rag.domain import Chunk
from local_rag.store import QdrantVectorStore


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
