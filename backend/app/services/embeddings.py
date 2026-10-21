"""Embedding model wrapper. The HuggingFace model is loaded lazily on first use."""
import logging
import threading

from app.core.errors import AppError

logger = logging.getLogger(__name__)


class HuggingFaceEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                logger.info("Loading embedding model %s (first run downloads it)...", self.model_name)
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings

                    self._model = HuggingFaceEmbeddings(
                        model_name=self.model_name,
                        encode_kwargs={"normalize_embeddings": True},
                    )
                except Exception as exc:
                    logger.exception("Embedding model failed to load")
                    raise AppError(
                        503,
                        "embedding_model_unavailable",
                        f"Could not load the embedding model '{self.model_name}'. "
                        "On first run it must be downloaded from HuggingFace, so check your internet connection "
                        f"and that requirements.txt is installed. Details: {type(exc).__name__}: {exc}",
                    ) from exc
                logger.info("Embedding model ready")
        return self._model

    def warmup(self) -> None:
        try:
            self._get_model()
        except AppError:
            pass  # reported again when it is actually needed

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._get_model().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._get_model().embed_query(text)
