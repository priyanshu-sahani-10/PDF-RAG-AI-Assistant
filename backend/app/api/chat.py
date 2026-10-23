from fastapi import APIRouter, Depends

from app.api.deps import get_services
from app.core.errors import AppError
from app.schemas.models import (
    ChatRequest,
    ChatResponse,
    DeleteOut,
    SessionCreate,
    SessionDetailOut,
    SessionListOut,
    SessionOut,
)
from app.services.container import Services

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(body: ChatRequest, services: Services = Depends(get_services)):
    return services.rag.answer(body.question, body.document_ids, body.session_id)


@router.get("/sessions", response_model=SessionListOut)
def list_sessions(services: Services = Depends(get_services)):
    return {"sessions": services.chat_store.list_sessions()}


@router.post("/sessions", response_model=SessionOut, status_code=201)
def create_session(body: SessionCreate | None = None, services: Services = Depends(get_services)):
    return services.chat_store.create_session(body.title if body else None)


@router.get("/sessions/{session_id}", response_model=SessionDetailOut)
def get_session(session_id: str, services: Services = Depends(get_services)):
    session = services.chat_store.get_session(session_id)
    if session is None:
        raise AppError(404, "session_not_found", "Chat session not found.")
    return {**session, "messages": services.chat_store.get_messages(session_id)}


@router.delete("/sessions/{session_id}", response_model=DeleteOut)
def delete_session(session_id: str, services: Services = Depends(get_services)):
    if not services.chat_store.delete_session(session_id):
        raise AppError(404, "session_not_found", "Chat session not found.")
    return {"deleted": True, "id": session_id}
