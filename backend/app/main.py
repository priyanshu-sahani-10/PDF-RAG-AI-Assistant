import logging
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, documents, health
from app.core.config import Settings, get_settings
from app.core.errors import register_error_handlers
from app.core.logging import setup_logging
from app.services.container import Services

logger = logging.getLogger(__name__)


def create_app(
    settings: Optional[Settings] = None,
    embedder: Optional[object] = None,
    ollama: Optional[object] = None,
) -> FastAPI:
    setup_logging()
    settings = settings or get_settings()
    services = Services(settings, embedder=embedder, ollama=ollama)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Load (and on first run download) the embedding model without blocking startup.
        if hasattr(services.embedder, "warmup"):
            threading.Thread(target=services.embedder.warmup, daemon=True).start()
        yield

    app = FastAPI(title="PDF RAG Chatbot API", version="1.0.0", lifespan=lifespan)
    app.state.services = services

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_error_handlers(app)
    app.include_router(health.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    return app
