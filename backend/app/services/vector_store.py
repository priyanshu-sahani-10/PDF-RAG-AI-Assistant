"""ChromaDB wrapper.

Distance semantics: the collection uses cosine space, so Chroma returns
distance = 1 - cosine_similarity (0 = identical, 2 = opposite). Embeddings are
normalised, so we report relevance = 1 - distance, clamped to [0, 1].
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document

COLLECTION_NAME = "pdf_chunks"
ADD_BATCH_SIZE = 128


@dataclass
class RetrievedChunk:
    text: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    distance: float

    @property
    def relevance(self) -> float:
        return max(0.0, min(1.0, 1.0 - self.distance))


class VectorStore:
    def __init__(self, path: Path):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list[Document], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        for start in range(0, len(chunks), ADD_BATCH_SIZE):
            batch = chunks[start : start + ADD_BATCH_SIZE]
            self.collection.add(
                ids=[f"{c.metadata['document_id']}:{c.metadata['chunk_index']}" for c in batch],
                documents=[c.page_content for c in batch],
                embeddings=embeddings[start : start + ADD_BATCH_SIZE],
                metadatas=[dict(c.metadata) for c in batch],
            )

    def search(
        self, query_embedding: list[float], top_k: int, document_ids: Optional[list[str]] = None
    ) -> list[RetrievedChunk]:
        """Nearest chunks, filtered inside the vector search itself."""
        where = None
        if document_ids is not None:
            if not document_ids:
                return []
            where = (
                {"document_id": document_ids[0]}
                if len(document_ids) == 1
                else {"document_id": {"$in": document_ids}}
            )
        total = self.collection.count()
        if total == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for text, meta, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            chunks.append(
                RetrievedChunk(
                    text=text,
                    document_id=meta["document_id"],
                    filename=meta["filename"],
                    page_number=int(meta["page_number"]),
                    chunk_index=int(meta["chunk_index"]),
                    distance=float(distance),
                )
            )
        return chunks

    def delete_document(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})

    def count_for_document(self, document_id: str) -> int:
        return len(self.collection.get(where={"document_id": document_id}, include=[])["ids"])
