import pytest

from app.services.pdf_processor import PdfProcessingError, extract_pages, split_pages
from tests.conftest import make_blank_pdf, make_encrypted_pdf, make_pdf


def test_extracts_text_with_page_numbers(tmp_path):
    path = tmp_path / "a.pdf"
    path.write_bytes(make_pdf(["Alpha page one", "Beta page two"]))
    pages, total = extract_pages(path)
    assert total == 2
    assert [p.metadata["page_number"] for p in pages] == [1, 2]
    assert "Alpha" in pages[0].page_content


def test_blank_pdf_reports_ocr_not_supported(tmp_path):
    path = tmp_path / "blank.pdf"
    path.write_bytes(make_blank_pdf())
    with pytest.raises(PdfProcessingError) as exc:
        extract_pages(path)
    assert exc.value.code == "no_text"
    assert "OCR" in exc.value.message


def test_corrupted_pdf(tmp_path):
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"%PDF-1.4 this is not really a pdf")
    with pytest.raises(PdfProcessingError) as exc:
        extract_pages(path)
    assert exc.value.code == "corrupted_pdf"


def test_password_protected_pdf(tmp_path):
    path = tmp_path / "locked.pdf"
    path.write_bytes(make_encrypted_pdf())
    with pytest.raises(PdfProcessingError) as exc:
        extract_pages(path)
    assert exc.value.code == "password_protected"


def test_chunks_have_required_metadata(tmp_path):
    path = tmp_path / "long.pdf"
    path.write_bytes(make_pdf(["word " * 30 + "\n" + "more text here " * 20, "second page content " * 10]))
    pages, _ = extract_pages(path)
    chunks = split_pages(pages, "docid", "long.pdf", chunk_size=200, chunk_overlap=30)
    assert len(chunks) > 2
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert c.metadata["document_id"] == "docid"
        assert c.metadata["filename"] == "long.pdf"
        assert c.metadata["page_number"] in (1, 2)
