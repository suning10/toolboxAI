from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Office Analytics Agent",
    description="Self-hosted LangChain agent over Ollama for internal data analytics",
    version="0.1.0",
)

app.include_router(router)

# Run with: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
