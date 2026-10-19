"""PDF text extraction and chunking."""
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)


class PdfProcessingError(Exception):
    """Raised with a message that is safe to show to the user."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def extract_pages(pdf_path: Path) -> tuple[list[Document], int]:
    """Return (pages with text, total page count). Page numbers are 1-based."""
    try:
        reader = PdfReader(str(pdf_path))
        if reader.is_encrypted:
            # Many PDFs are "encrypted" with an empty user password.
            if not reader.decrypt(""):
                raise PdfProcessingError(
                    "password_protected",
                    "This PDF is password protected. Remove the password and upload it again.",
                )
        total_pages = len(reader.pages)
    except PdfProcessingError:
        raise
    except (PdfReadError, ValueError, KeyError, OSError, TypeError) as exc:
        logger.warning("Could not read PDF %s: %s", pdf_path.name, exc)
        raise PdfProcessingError("corrupted_pdf", "This file could not be read. It may be corrupted.") from exc
    except Exception as exc:  # pypdf can raise odd errors on broken files
        logger.warning("Unexpected PDF error for %s: %s", pdf_path.name, exc)
        raise PdfProcessingError("corrupted_pdf", "This file could not be read. It may be corrupted.") from exc

    if total_pages == 0:
        raise PdfProcessingError("empty_pdf", "This PDF has no pages.")

    pages: list[Document] = []
    for index, page in enumerate(reader.pages):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as exc:
            logger.warning("Skipping page %s of %s: %s", index + 1, pdf_path.name, exc)
            continue
        if text:
            pages.append(Document(page_content=text, metadata={"page_number": index + 1}))

    if not pages:
        raise PdfProcessingError(
            "no_text",
            "No text could be extracted. This looks like a scanned or image-only PDF. "
            "OCR is not supported in this version.",
        )
    return pages, total_pages


def split_pages(
    pages: list[Document],
    document_id: str,
    filename: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split pages into chunks. Every chunk carries the metadata needed for retrieval and citations."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(pages)
    result: list[Document] = []
    for index, chunk in enumerate(chunks):
        text = chunk.page_content.strip()
        if not text:
            continue
        result.append(
            Document(
                page_content=text,
                metadata={
                    "document_id": document_id,
                    "filename": filename,
                    "page_number": int(chunk.metadata["page_number"]),
                    "chunk_index": len(result),
                },
            )
        )
    return result
