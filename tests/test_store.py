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
    )

    store.upsert([chunk], [[1.0, 0.0]])
    results = store.search([1.0, 0.0], limit=1, threshold=0.1)

    assert len(results) == 1
    assert results[0].text == "Local evidence"
    assert results[0].score == pytest.approx(1.0)

    store.clear()
    assert store.search([1.0, 0.0], limit=1, threshold=0.1) == []
