from fastapi import APIRouter

from app.api.v1.endpoints import chat, sops, upload

api_router = APIRouter(prefix="/admin", tags=["admins"])
api_router.include_router(upload.router, tags=["datasets"])
api_router.include_router(sops.router, tags=["knowledge"])
api_router.include_router(chat.router, tags=["chat"])
