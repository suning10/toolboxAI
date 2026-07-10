import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel

from app.config import DATA_UPLOAD_DIR
from app.agent.memory import build_agent_with_memory, invoke_with_history
from app.tools.pandas_tool import load_dataset
from app.tools.sql_tool import load_dataset_sql
from app.api.auth import verify_api_key

router = APIRouter()

# Built once at import time, reused across requests - avoids reloading
# the model connection on every call. Fine at your call volume; if you
# scale up, consider a connection pool per worker instead.
_agent = build_agent_with_memory()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/upload")
def upload_file(file: UploadFile = File(...), _=Depends(verify_api_key)):
    os.makedirs(DATA_UPLOAD_DIR, exist_ok=True)
    filepath = os.path.join(DATA_UPLOAD_DIR, file.filename)

    with open(filepath, "wb") as f:
        f.write(file.file.read())

    if not filepath.endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(400, "Only .csv, .xlsx, .xls files are supported")

    pandas_msg = load_dataset(filepath)
    sql_msg = load_dataset_sql(filepath)
    return {"status": "loaded", "detail": f"{pandas_msg} | {sql_msg}"}


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, _=Depends(verify_api_key)):
    session_id = req.session_id or str(uuid.uuid4())
    try:
        answer = invoke_with_history(_agent, req.message, session_id)
    except Exception as e:
        raise HTTPException(500, f"Agent error: {e}")
    return ChatResponse(response=answer, session_id=session_id)
