from fastapi import APIRouter, Depends

from app.api.deps import get_services
from app.schemas.models import HealthOut
from app.services.container import Services

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
def health(services: Services = Depends(get_services)):
    settings = services.settings
    return {
        "status": "ok",
        "backend": True,
        "ollama": services.ollama.status(),
        "embedding_model": settings.embedding_model,
        "max_upload_mb": settings.max_upload_mb,
        "top_k": settings.top_k,
        "relevance_threshold": settings.relevance_threshold,
    }
