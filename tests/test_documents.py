import pytest

from local_rag.documents import chunk_pages, load_document
from local_rag.domain import Page, RAGError


@pytest.mark.unit
def test_chunks_are_overlapping_and_stable() -> None:
    pages = [Page("notes.txt", 1, "one two three four five six")]

    first = chunk_pages(pages, chunk_size=12, overlap=4)
    second = chunk_pages(pages, chunk_size=12, overlap=4)

    assert len(first) == 3
    assert first == second
    assert first[0].text[-4:] == first[1].text[:4]


@pytest.mark.unit
def test_rejects_unsupported_documents() -> None:
    with pytest.raises(RAGError, match="Unsupported"):
        load_document("notes.docx", b"content")
