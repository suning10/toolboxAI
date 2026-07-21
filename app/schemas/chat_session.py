from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ChatSessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    message_count: int


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatSessionMessages(BaseModel):
    session_id: str
    messages: list[ChatMessage]
