from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ChatSessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime
    updated_at: datetime
    message_count: int
