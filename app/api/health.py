"""
Unversioned liveness check, mounted directly at /health in app/main.py.
Kept outside /api/v1 since load balancers and container orchestrators
expect a stable, version-independent health path.
"""
from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness check")
def health():
    return {"status": "ok"}
