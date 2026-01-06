from fastapi import APIRouter

from app.api.routes import rag, dev

api_router = APIRouter()

api_router.include_router(rag.router)
api_router.include_router(dev.router)