from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict
from chromadb.config import Settings

class ChromaSettings(Settings):

    class Config:
        env_file = ".env"
        extra = "ignore"
        env_file_encoding = "utf-8"
        
    is_persistent: bool
    persist_directory: str
    anonymized_telemetry: bool = False
    allow_reset: bool = False

class ProjectSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: ClassVar[str] = "Wukong Legal AI"
    llama_model_path: str
    cors_origins: list[str]
    trusted_hosts: list[str]
    embed_model_name: str
    embed_model_collection: str
    sqlite_filepath: str = "conversations.db"

chromasettings = ChromaSettings()
settings = ProjectSettings()