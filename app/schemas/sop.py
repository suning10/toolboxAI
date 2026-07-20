from pydantic import BaseModel


class SopIngestResponse(BaseModel):
    status: str
    filename: str
    detail: str
