"""Upload validation, background indexing and deletion of documents."""
import hashlib
import logging
import os
import re
import threading
from pathlib import Path
from typing import BinaryIO

from app.core.config import Settings
from app.core.errors import AppError
from app.services.document_store import DocumentStore
from app.services.pdf_processor import PdfProcessingError, extract_pages, split_pages
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {"application/pdf", "application/x-pdf", "application/octet-stream", "", None}
READ_CHUNK = 1024 * 1024
EMBED_BATCH = 64


def sanitize_filename(name: str) -> str:
    """Keep only a safe display name. The file on disk is never named after user input."""
    name = (name or "").replace("\\", "/").split("/")[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)
    name = re.sub(r"[^\w\-. ()\[\]]", "_", name).strip(" .")
    if not name.lower().endswith(".pdf"):
        name = (name or "document") + ".pdf"
    stem, ext = name[:-4], name[-4:]
    return stem[:120] + ext


class DocumentService:
    def __init__(self, settings: Settings, store: DocumentStore, vectors: VectorStore, embedder, lock: threading.RLock):
        self.settings = settings
        self.store = store
        self.vectors = vectors
        self.embedder = embedder
        self.lock = lock  # serialises every write to Chroma + SQLite
        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = self.settings.upload_dir / ".tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    # ---------- upload ----------
    def save_upload(self, filename: str, content_type: str | None, stream: BinaryIO) -> dict:
        """Validate and store the PDF, register it as 'processing'. Indexing happens in index_document()."""
        clean_name = sanitize_filename(filename)
        if not (filename or "").lower().endswith(".pdf"):
            raise AppError(400, "invalid_file_type", "Only PDF files (.pdf) are supported.")
        if (content_type or "").split(";")[0].strip().lower() not in ALLOWED_CONTENT_TYPES:
            raise AppError(400, "invalid_file_type", "Only PDF files are supported.")

        tmp_path = self.tmp_dir / f"{os.getpid()}-{threading.get_ident()}-{os.urandom(4).hex()}.part"
        digest = hashlib.sha256()
        size = 0
        try:
            with open(tmp_path, "wb") as out:
                first = True
                while True:
                    block = stream.read(READ_CHUNK)
                    if not block:
                        break
                    if first:
                        if b"%PDF-" not in block[:1024]:
                            raise AppError(400, "invalid_file_type", "This file is not a valid PDF.")
                        first = False
                    size += len(block)
                    if size > self.settings.max_upload_bytes:
                        raise AppError(
                            413,
                            "file_too_large",
                            f"File is larger than the {self.settings.max_upload_mb} MB limit.",
                        )
                    digest.update(block)
                    out.write(block)
            if size == 0:
                raise AppError(400, "empty_file", "The uploaded file is empty.")

            doc_id = digest.hexdigest()[:16]
            stored_name = f"{doc_id}.pdf"
            with self.lock:
                existing = self.store.get(doc_id)
                if existing and existing["status"] != "failed":
                    raise AppError(
                        409,
                        "duplicate_document",
                        f"'{existing['filename']}' has already been uploaded.",
                        details={"document_id": doc_id},
                    )
                if existing:  # a failed attempt: clear it and try again
                    self._remove_everything(doc_id)
                os.replace(tmp_path, self.settings.upload_dir / stored_name)
                if not self.store.create(doc_id, clean_name, stored_name, size):
                    raise AppError(409, "duplicate_document", "This file is already being processed.")
            return self.store.get(doc_id)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    # ---------- indexing (runs in a background thread) ----------
    def index_document(self, doc_id: str) -> None:
        with self.lock:
            doc = self.store.get(doc_id)
            if doc is None:  # deleted while waiting in the queue
                return
            pdf_path = self.settings.upload_dir / doc["stored_name"]
            try:
                pages, total_pages = extract_pages(pdf_path)
                chunks = split_pages(
                    pages, doc_id, doc["filename"], self.settings.chunk_size, self.settings.chunk_overlap
                )
                if not chunks:
                    raise PdfProcessingError("no_text", "No text could be extracted from this PDF.")
                texts = [c.page_content for c in chunks]
                embeddings: list[list[float]] = []
                for start in range(0, len(texts), EMBED_BATCH):
                    embeddings.extend(self.embedder.embed_documents(texts[start : start + EMBED_BATCH]))
                self.vectors.add_chunks(chunks, embeddings)
                self.store.mark_ready(doc_id, total_pages, len(chunks))
                logger.info("Indexed %s: %d pages, %d chunks", doc["filename"], total_pages, len(chunks))
            except PdfProcessingError as exc:
                self._fail(doc_id, pdf_path, exc.message)
            except AppError as exc:
                self._fail(doc_id, pdf_path, exc.message)
            except Exception:
                logger.exception("Indexing failed for %s", doc_id)
                self._fail(doc_id, pdf_path, "Indexing failed unexpectedly. Check the backend logs.")

    def _fail(self, doc_id: str, pdf_path: Path, message: str) -> None:
        """Roll back partial work so a failed document never looks indexed."""
        try:
            self.vectors.delete_document(doc_id)
        except Exception:
            logger.exception("Could not clean vectors for %s", doc_id)
        pdf_path.unlink(missing_ok=True)
        self.store.mark_failed(doc_id, message)

    # ---------- read / delete ----------
    def get(self, doc_id: str) -> dict:
        doc = self.store.get(doc_id)
        if doc is None:
            raise AppError(404, "document_not_found", "Document not found.")
        return doc

    def list_all(self) -> list[dict]:
        return self.store.list_all()

    def delete(self, doc_id: str) -> None:
        with self.lock:
            if self.store.get(doc_id) is None:
                raise AppError(404, "document_not_found", "Document not found.")
            self._remove_everything(doc_id)

    def _remove_everything(self, doc_id: str) -> None:
        doc = self.store.get(doc_id)
        self.vectors.delete_document(doc_id)
        if doc:
            (self.settings.upload_dir / doc["stored_name"]).unlink(missing_ok=True)
        self.store.delete(doc_id)
