from pydantic_settings import BaseSettings, SettingsConfigDict
# from chromadb.config import Settings

class ChromaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    persist_directory: str = "./chroma_db"
    anonymized_telemetry: bool = False
    allow_reset: bool = False

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "GenAI"
    llama_model_path: str
    cors_origins: list[str]
    trusted_hosts: list[str]

chromasettings = ChromaSettings()
settings = Settings()