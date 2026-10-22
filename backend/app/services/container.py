"""Builds every service once. Tests can pass fake embedders / Ollama clients."""
import threading
from typing import Optional

from app.core.config import Settings
from app.database.db import Database
from app.services.chat_store import ChatStore
from app.services.document_service import DocumentService
from app.services.document_store import DocumentStore
from app.services.embeddings import HuggingFaceEmbedder
from app.services.ollama_client import OllamaClient
from app.services.rag_service import RagService
from app.services.vector_store import VectorStore


class Services:
    def __init__(self, settings: Settings, embedder: Optional[object] = None, ollama: Optional[object] = None):
        self.settings = settings
        self.lock = threading.RLock()
        self.db = Database(settings.sqlite_file)
        self.embedder = embedder or HuggingFaceEmbedder(settings.embedding_model)
        self.ollama = ollama or OllamaClient(
            settings.ollama_base_url,
            settings.ollama_model,
            settings.ollama_timeout_seconds,
            settings.ollama_health_timeout_seconds,
        )
        self.vectors = VectorStore(settings.chroma_path)
        self.document_store = DocumentStore(self.db)
        self.chat_store = ChatStore(self.db)
        self.document_store.fail_interrupted()
        self.documents = DocumentService(settings, self.document_store, self.vectors, self.embedder, self.lock)
        self.rag = RagService(settings, self.document_store, self.chat_store, self.vectors, self.embedder, self.ollama)
