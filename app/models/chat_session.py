"""
Metadata about a chat session, keyed by the same session_id LangGraph
uses as its thread_id. The actual conversation turns live in the
checkpointer's own tables (app/agent/memory.py, SqliteSaver) - this
table exists so the API can list/inspect sessions without deserializing
LangGraph's internal checkpoint format.
"""
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
