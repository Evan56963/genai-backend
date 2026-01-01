from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager

from app.api.register import api_router
from app.config import settings
from app.llama_loader import load_llama_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load llama and save to app state
    app.state.llama = load_llama_model(settings.llama_model_path)

    yield

    # Cleanup if necessary
    del app.state.llama

app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost"]
)

app.include_router(api_router)