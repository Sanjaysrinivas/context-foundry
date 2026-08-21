import pymupdf
import pytest

from local_rag.documents import chunk_pages, load_document
from local_rag.domain import Page, RAGError


@pytest.mark.unit
def test_chunks_are_overlapping_and_stable() -> None:
    pages = [
        Page(
            "notes.md",
            1,
            "# Skills\n\nPython PostgreSQL Docker\n\n## Experience\n\nBuilt local AI systems",
        )
    ]

    first = chunk_pages(pages, chunk_size=30, overlap=5)
    second = chunk_pages(pages, chunk_size=30, overlap=5)

    assert first == second
    renamed = chunk_pages([Page("other.md", 1, pages[0].text)], chunk_size=30, overlap=5)
    assert first[0].document_id != renamed[0].document_id
    assert all(len(chunk.text) <= 30 for chunk in first)
    assert any(chunk.text.startswith("# Skills") for chunk in first)
    assert any("PostgreSQL" in chunk.text for chunk in first)


@pytest.mark.unit
def test_pdf_extraction_preserves_pages_and_headings() -> None:
    document = pymupdf.open()  # type: ignore[no-untyped-call]
    first_page = document.new_page()
    first_page.insert_text((72, 72), "SKILLS", fontsize=18)
    first_page.insert_text((72, 105), "Python, PostgreSQL, AWS and Docker")
    second_page = document.new_page()
    second_page.insert_text((72, 72), "EXPERIENCE", fontsize=18)
    second_page.insert_text((72, 105), "Built local retrieval systems")
    content = document.tobytes()  # type: ignore[no-untyped-call]
    document.close()  # type: ignore[no-untyped-call]

    pages = load_document("resume.pdf", content)

    assert [page.number for page in pages] == [1, 2]
    assert "# SKILLS" in pages[0].text
    assert "Python, PostgreSQL, AWS and Docker" in pages[0].text
    assert "# EXPERIENCE" in pages[1].text


@pytest.mark.unit
def test_pdf_extraction_uses_ocr_for_image_only_pages() -> None:
    source = pymupdf.open()  # type: ignore[no-untyped-call]
    source_page = source.new_page()
    source_page.insert_text((72, 72), "SCANNED SKILLS: Python and AWS", fontsize=16)
    image = source_page.get_pixmap(matrix=pymupdf.Matrix(2, 2))  # type: ignore[no-untyped-call]

    scanned = pymupdf.open()  # type: ignore[no-untyped-call]
    scanned_page = scanned.new_page(width=source_page.rect.width, height=source_page.rect.height)
    image_bytes = image.tobytes("png")  # type: ignore[no-untyped-call]
    scanned_page.insert_image(  # type: ignore[no-untyped-call]
        scanned_page.rect, stream=image_bytes
    )
    content = scanned.tobytes()  # type: ignore[no-untyped-call]
    source.close()  # type: ignore[no-untyped-call]
    scanned.close()  # type: ignore[no-untyped-call]

    pages = load_document("scan.pdf", content)

    assert "SCANNED SKILLS: Python and AWS" in pages[0].text


@pytest.mark.unit
def test_rejects_unsupported_documents() -> None:
    with pytest.raises(RAGError, match="Unsupported"):
        load_document("notes.docx", b"content")
