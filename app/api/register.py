from fastapi import APIRouter

from app.api.routes import rag, manage

api_router = APIRouter()

api_router.include_router(rag.router)
api_router.include_router(manage.router)