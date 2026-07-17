from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.exceptions import register_exception_handlers
from app.api.health import router as health_router
from app.api.v1.router import api_router
from app.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Office Analytics Agent",
    description="Self-hosted LangChain agent over Ollama for internal data analytics",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(api_router, prefix="/api/v1")
register_exception_handlers(app)

# Run with: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
