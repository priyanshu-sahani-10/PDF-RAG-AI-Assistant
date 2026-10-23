from typing import Optional

from pydantic import BaseModel, Field, field_validator


class DocumentOut(BaseModel):
    id: str
    filename: str
    status: str  # processing | ready | failed
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    size_bytes: int
    error: Optional[str] = None
    created_at: str


class DocumentListOut(BaseModel):
    documents: list[DocumentOut]


class DeleteOut(BaseModel):
    deleted: bool
    id: str


class OllamaStatus(BaseModel):
    available: bool
    model: str
    model_available: bool
    message: str


class HealthOut(BaseModel):
    status: str
    backend: bool
    ollama: OllamaStatus
    embedding_model: str
    max_upload_mb: int
    top_k: int
    relevance_threshold: float


class SourceOut(BaseModel):
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    excerpt: str
    text: str
    relevance: float


class ChatRequest(BaseModel):
    question: str = Field(..., max_length=2000)
    document_ids: Optional[list[str]] = None
    session_id: Optional[str] = None

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be empty")
        return value


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    session_id: str
    grounded: bool  # False when nothing relevant was found in the documents


class SessionCreate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=120)


class SessionOut(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class SessionListOut(BaseModel):
    sessions: list[SessionOut]


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    sources: list[SourceOut] = []
    created_at: str


class SessionDetailOut(SessionOut):
    messages: list[MessageOut]
