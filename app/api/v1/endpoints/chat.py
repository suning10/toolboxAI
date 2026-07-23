import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.agent.memory import (
    build_agent_with_memory,
    get_message_history,
    invoke_with_history,
    stream_with_history,
)
from app.api.deps import verify_api_key
from app.config import INVENTORY_API_BASE_URL
from app.db.session import get_db
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.chat_session import ChatSessionMessages, ChatSessionRead
from app.services import chat_session_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai")

# Built once at import time, reused across requests - avoids reloading
# the model connection on every call. Fine at your call volume; if you
# scale up, consider a connection pool per worker instead.
_agent = build_agent_with_memory()


@router.post("/chat", response_model=ChatResponse, summary="Ask a question about SCR Report")
def chat(req: ChatRequest, db: Session = Depends(get_db)): #, _=Depends(verify_api_key)):
    session_id = req.session_id or str(uuid.uuid4())
    answer = invoke_with_history(_agent, req.message, session_id)
    chat_session_service.touch_session(db, session_id, first_message=req.message)
    return ChatResponse(response=answer, session_id=session_id)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/chat/stream", summary="Ask a question, streamed token-by-token via SSE")
def chat_stream(req: ChatRequest, db: Session = Depends(get_db)):
    session_id = req.session_id or str(uuid.uuid4())

    def gen():
        yield _sse("session", {"session_id": session_id})
        chunks: list[str] = []
        try:
            for token in stream_with_history(_agent, req.message, session_id):
                chunks.append(token)
                yield _sse("token", {"content": token})
        except Exception as e:
            logger.exception("Streaming chat failed for session %s", session_id)
            yield _sse("error", {"detail": str(e)})
            return
        finally:
            chat_session_service.touch_session(db, session_id, first_message=req.message)
        yield _sse("done", {"response": "".join(chunks)})

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get(
    "/chat/sessions",
    response_model=list[ChatSessionRead],
    summary="List known chat sessions",
)
def list_chat_sessions(db: Session = Depends(get_db)): # , _=Depends(verify_api_key)):
    return chat_session_service.list_sessions(db)


@router.get(
    "/chat/sessions/{session_id}",
    response_model=ChatSessionRead,
    summary="Get one chat session's metadata",
)
def get_chat_session(session_id: str, db: Session = Depends(get_db)):#, _=Depends(verify_api_key)):
    session = chat_session_service.get_session(db, session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    return session


@router.get(
    "/chat/sessions/{session_id}/messages",
    response_model=ChatSessionMessages,
    summary="Get the full message history for one chat session",
)
def get_chat_session_messages(
    session_id: str, db: Session = Depends(get_db) #, _=Depends(verify_api_key)
):
    if chat_session_service.get_session(db, session_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    messages = get_message_history(_agent, session_id)
    return ChatSessionMessages(session_id=session_id, messages=messages)
