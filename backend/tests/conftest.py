import hashlib
import io
import math
import re

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from app.core.config import Settings
from app.core.errors import AppError
from app.main import create_app

DIM = 512


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder (normalised). No downloads needed."""

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


class FakeOllama:
    def __init__(self, available=True, model_available=True):
        self.available = available
        self.model_available = model_available
        self.calls = []

    def status(self):
        return {
            "available": self.available,
            "model": "gemma:2b",
            "model_available": self.model_available,
            "message": "Ollama is ready." if self.available and self.model_available else "not ready",
        }

    def generate(self, system_prompt, user_prompt):
        if not self.available:
            raise AppError(503, "ollama_unavailable", "Ollama is not reachable.")
        if not self.model_available:
            raise AppError(503, "model_missing", "Model not downloaded.")
        self.calls.append((system_prompt, user_prompt))
        return "Fake answer based on the context."


def make_pdf(pages: list[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for text in pages:
        y = 740
        for line in text.split("\n"):
            c.drawString(60, y, line)
            y -= 16
        c.showPage()
    c.save()
    return buf.getvalue()


def make_blank_pdf() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.showPage()
    c.save()
    return buf.getvalue()


def make_encrypted_pdf(password="secret") -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt(password)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path):
    return Settings(
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "uploads",
        max_upload_mb=1,
        chunk_size=300,
        chunk_overlap=50,
        top_k=4,
        relevance_threshold=0.2,
    )


@pytest.fixture
def ollama():
    return FakeOllama()


@pytest.fixture
def client(settings, ollama):
    app = create_app(settings=settings, embedder=FakeEmbedder(), ollama=ollama)
    return TestClient(app)


def upload(client, data: bytes, name="doc.pdf", content_type="application/pdf"):
    return client.post("/api/documents/upload", files={"file": (name, data, content_type)})
