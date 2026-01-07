from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.register import api_router
from app.config import settings, chromasettings
from app.initial_data import load_llama_model, initialize_chromadb_client, initialize_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.tokenizer = await run_in_threadpool(load_llama_model, settings.llama_model_path)
    app.state.embed_model, app.state.collection = await run_in_threadpool(initialize_chromadb_client,
        
        settings.embed_model_name,
        chromasettings.persist_directory,
        settings.embed_model_collection

    )

    app.state.session = initialize_db(settings.sqlite_filepath)

    yield

    del app.state.model
    del app.state.tokenizer
    del app.state.embed_model
    del app.state.collection
    del app.state.session
    
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