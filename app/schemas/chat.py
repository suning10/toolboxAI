from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message to chat")
    session_id: str | None = Field(
        None, description="Existing session to continue; a new one is created if omitted"
    )
    include_reasoning: bool = Field(
        False,
        description="Streaming only: also emit the model's reasoning/thinking as "
        "'reasoning' SSE events. Ignored by the non-streaming /chat endpoint.",
    )


class ChatResponse(BaseModel):
    response: str
    session_id: str
