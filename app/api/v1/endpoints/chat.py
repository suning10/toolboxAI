import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.agent.memory import build_agent_with_memory, invoke_with_history
from app.api.deps import verify_api_key
from app.db.session import get_db
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.chat_session import ChatSessionRead
from app.services import chat_session_service

router = APIRouter()

# Built once at import time, reused across requests - avoids reloading
# the model connection on every call. Fine at your call volume; if you
# scale up, consider a connection pool per worker instead.
_agent = build_agent_with_memory()


@router.post("/chat", response_model=ChatResponse, summary="Ask a question about the loaded dataset")
def chat(req: ChatRequest, db: Session = Depends(get_db), _=Depends(verify_api_key)):
    session_id = req.session_id or str(uuid.uuid4())
    answer = invoke_with_history(_agent, req.message, session_id)
    chat_session_service.touch_session(db, session_id)
    return ChatResponse(response=answer, session_id=session_id)


@router.get(
    "/chat/sessions",
    response_model=list[ChatSessionRead],
    summary="List known chat sessions",
)
def list_chat_sessions(db: Session = Depends(get_db), _=Depends(verify_api_key)):
    return chat_session_service.list_sessions(db)


@router.get(
    "/chat/sessions/{session_id}",
    response_model=ChatSessionRead,
    summary="Get one chat session's metadata",
)
def get_chat_session(session_id: str, db: Session = Depends(get_db), _=Depends(verify_api_key)):
    session = chat_session_service.get_session(db, session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    return session
