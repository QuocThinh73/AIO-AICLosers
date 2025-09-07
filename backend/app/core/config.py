from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Dict, Union
from uuid import UUID
from functools import lru_cache


class Settings(BaseSettings):
    API_V1_STR: str
    PROJECT_NAME: str
    
    QDRANT_HOST: str
    QDRANT_PORT: int

    OPENCLIP_COLLECTION_NAME: str
    CAPTION_COLLECTION_NAME: str

    UUID_NAMESPACE: UUID

    # Load environment variables
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
