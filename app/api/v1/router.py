from fastapi import APIRouter

from app.api.v1.endpoints import chat, upload

api_router = APIRouter()
api_router.include_router(upload.router, tags=["datasets"])
api_router.include_router(chat.router, tags=["chat"])
