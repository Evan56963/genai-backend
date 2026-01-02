from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager

from app.api.register import api_router
from app.config import settings
from app.llama_loader import load_llama_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llama = load_llama_model(settings.llama_model_path)

    yield

    del app.state.llama

app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.trusted_hosts,
)

app.include_router(api_router)