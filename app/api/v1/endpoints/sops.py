import os

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.deps import verify_api_key
from app.config import SOP_DOCS_DIR
from app.knowledge.ingest import ALLOWED_EXTENSIONS, ingest_sop_file
from app.schemas.sop import SopIngestResponse

router = APIRouter()


@router.post(
    "/sops/upload",
    response_model=SopIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest an SOP document into the knowledge base",
)
def upload_sop(file: UploadFile = File(...), _=Depends(verify_api_key)):
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Only {', '.join(ALLOWED_EXTENSIONS)} files are supported",
        )

    os.makedirs(SOP_DOCS_DIR, exist_ok=True)
    filepath = os.path.join(SOP_DOCS_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(file.file.read())

    detail = ingest_sop_file(filepath)
    return SopIngestResponse(status="ingested", filename=filename, detail=detail)
