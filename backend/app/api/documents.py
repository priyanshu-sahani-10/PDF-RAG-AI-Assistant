from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile

from app.api.deps import get_services
from app.schemas.models import DeleteOut, DocumentListOut, DocumentOut
from app.services.container import Services

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentOut, status_code=202)
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    services: Services = Depends(get_services),
):
    """Validate + store the PDF, then index it in the background. Poll GET /documents for the status."""
    document = services.documents.save_upload(file.filename or "", file.content_type, file.file)
    background_tasks.add_task(services.documents.index_document, document["id"])
    return document


@router.get("", response_model=DocumentListOut)
def list_documents(services: Services = Depends(get_services)):
    return {"documents": services.documents.list_all()}


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, services: Services = Depends(get_services)):
    return services.documents.get(document_id)


@router.delete("/{document_id}", response_model=DeleteOut)
def delete_document(document_id: str, services: Services = Depends(get_services)):
    services.documents.delete(document_id)
    return {"deleted": True, "id": document_id}
