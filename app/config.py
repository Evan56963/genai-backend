from pydantic_settings import BaseSettings, SettingsConfigDict
from chromadb.config import Settings

class ChromaSettings(Settings):

    class Config:
        env_file = ".env"
        extra = "ignore"
        env_file_encoding = "utf-8"

    persist_directory: str = "./chroma_db"
    anonymized_telemetry: bool = False
    allow_reset: bool = False

class ProjectSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "GenAI"
    llama_model_path: str
    cors_origins: list[str]
    trusted_hosts: list[str]

chromasettings = ChromaSettings()
settings = ProjectSettings()