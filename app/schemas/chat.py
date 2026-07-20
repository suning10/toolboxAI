from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message to chat")
    session_id: str | None = Field(
        None, description="Existing session to continue; a new one is created if omitted"
    )


class ChatResponse(BaseModel):
    response: str
    session_id: str
