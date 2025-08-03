import os
import torch
from typing import Dict, List, Any, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
import json

class Settings(BaseSettings):
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "AIO-AIClosers"
    
    # Application folders
    UPLOAD_FOLDER: str
    DATABASE_FOLDER: str
    
    # Qdrant configuration
    QDRANT_HOST: str
    QDRANT_PORT: int
    CAPTIONS_COLLECTION: str
    
    # Elasticsearch configuration
    ELASTICSEARCH_HOSTS: List[str]
    OCR_INDEX: str
    OBJECT_DETECTION_INDEX: str
    
    # CORS settings
    BACKEND_CORS_ORIGINS: Union[List[str], str]
    
    # Các trường được tính toán hoặc tham chiếu từ .env
    DEVICE: str
    KEYFRAMES_FOLDER: str
    SHOTS_FOLDER: str
    VIDEOS_FOLDER: str
    EMBEDDING_FOLDER: str
    MAPPING_JSON: str
    
    # Available embedding models configuration
    EMBEDDING_MODELS: Dict[str, Dict[str, str]] = {
        "OpenCLIP ViT-B-16 dfn2b": {
            "model_type": "openclip",
            "backbone": "ViT-B-16", 
            "pretrained": "dfn2b",
            "embeddings_file": "OpenCLIP_ViT-B-16_dfn2b_embeddings.bin"
        },
        "OpenCLIP ViT-B-16 webli": {
            "model_type": "openclip",
            "backbone": "ViT-B-16", 
            "pretrained": "webli",
            "embeddings_file": "OpenCLIP_ViT-B-16_webli_embeddings.bin"
        },
    }
    
    
    model_config = SettingsConfigDict(env_file="backend/.env", env_file_encoding="utf-8", case_sensitive=True)
    
    def model_post_init(self, __context):
        # Configure device based on CUDA availability
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Parse BACKEND_CORS_ORIGINS as JSON if it's a string
        if isinstance(self.BACKEND_CORS_ORIGINS, str):
            try:
                self.BACKEND_CORS_ORIGINS = json.loads(self.BACKEND_CORS_ORIGINS)
            except json.JSONDecodeError:
                # Fallback to a default if parsing fails
                self.BACKEND_CORS_ORIGINS = ["http://localhost"]
                
        # Set up database paths if not provided in .env
        if not self.KEYFRAMES_FOLDER:
            self.KEYFRAMES_FOLDER = os.path.join(self.DATABASE_FOLDER, "keyframes")
        if not self.SHOTS_FOLDER:
            self.SHOTS_FOLDER = os.path.join(self.DATABASE_FOLDER, "shots")
        if not self.VIDEOS_FOLDER:
            self.VIDEOS_FOLDER = os.path.join(self.DATABASE_FOLDER, "videos")
        if not self.EMBEDDING_FOLDER:
            self.EMBEDDING_FOLDER = os.path.join(self.DATABASE_FOLDER, "embeddings")
        if not self.MAPPING_JSON:
            self.MAPPING_JSON = os.path.join(self.DATABASE_FOLDER, "id2path.json")
        
    def get_embedding_model_names(self) -> List[str]:
        """Get list of available embedding model names."""
        return list(self.EMBEDDING_MODELS.keys())

@lru_cache()
def get_settings() -> Settings:
    return Settings()

# Export settings instance
settings = get_settings()

# Legacy constants (direct access for backward compatibility)
DEVICE = settings.DEVICE
UPLOAD_FOLDER = settings.UPLOAD_FOLDER
DATABASE_FOLDER = settings.DATABASE_FOLDER
KEYFRAMES_FOLDER = settings.KEYFRAMES_FOLDER
SHOTS_FOLDER = settings.SHOTS_FOLDER
VIDEOS_FOLDER = settings.VIDEOS_FOLDER
EMBEDDING_FOLDER = settings.EMBEDDING_FOLDER
MAPPING_JSON = settings.MAPPING_JSON
EMBEDDING_MODELS = settings.EMBEDDING_MODELS
CAPTIONS_COLLECTION = settings.CAPTIONS_COLLECTION
QDRANT_HOST = settings.QDRANT_HOST
QDRANT_PORT = settings.QDRANT_PORT
ELASTICSEARCH_HOSTS = settings.ELASTICSEARCH_HOSTS
OCR_INDEX = settings.OCR_INDEX
OBJECT_DETECTION_INDEX = settings.OBJECT_DETECTION_INDEX
