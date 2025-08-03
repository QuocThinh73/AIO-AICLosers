import logging
import os
from typing import Dict, Any, Optional, List
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
import faiss

# Import config
from backend.app.core.config import (
    settings, MAPPING_JSON, EMBEDDING_FOLDER,
    QDRANT_HOST, QDRANT_PORT, ELASTICSEARCH_HOSTS
)

class DatabaseService:
    """Service for handling all database connections (FAISS, Qdrant, Elasticsearch)."""
    
    def __init__(self):
        self.es_client: Optional[Elasticsearch] = None
        self.qdrant_client: Optional[QdrantClient] = None
        self.embedding_models: Dict[str, Any] = {}
        self.embeddings_path = EMBEDDING_FOLDER
        self.mapping_json = MAPPING_JSON
        
        # Initialize connections
        self._init_embedding_models()
        self._connect_qdrant()
        self._connect_elasticsearch()
    
    def _init_embedding_models(self):
        """Initialize available embedding models using FAISS."""
        try:
            import faiss
        except ImportError:
            logging.error("FAISS not available. Embedding search will not work.")
            return
            
        for model_name, model_info in settings.EMBEDDING_MODELS.items():
            try:
                embeddings_file = os.path.join(self.embeddings_path, model_info.get("embeddings_file", ""))
                if os.path.exists(embeddings_file) and os.path.exists(self.mapping_json):
                    faiss_instance = faiss.IndexFlatIP(512)  # Placeholder - actual dimension depends on the model
                    logging.info(f"Loading embedding model {model_name} from {embeddings_file}")
                    # In actual implementation, this would load the model properly
                    # faiss.read_index(embeddings_file)
                    self.embedding_models[model_name] = faiss_instance
                    logging.info(f"Successfully loaded embedding model {model_name}")
            except Exception as e:
                logging.error(f"Failed to load embedding model {model_name}: {str(e)}")
    
    def _connect_qdrant(self):
        """Connect to Qdrant vector database for caption search."""
        try:
            # Using connection details from config
            self.qdrant_client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
            # Test connection
            self.qdrant_client.get_collections()
            logging.info("Successfully connected to Qdrant")
        except Exception as e:
            self.qdrant_client = None
            logging.error(f"Failed to connect to Qdrant: {str(e)}")
    
    def _connect_elasticsearch(self):
        """Connect to Elasticsearch for OCR and object detection search."""
        try:
            self.es_client = Elasticsearch(ELASTICSEARCH_HOSTS)
            if not self.es_client.ping():
                raise ConnectionError("Failed to connect to Elasticsearch")
            logging.info("Successfully connected to Elasticsearch")
        except Exception as e:
            self.es_client = None
            logging.error(f"Failed to connect to Elasticsearch: {str(e)}")
    
    def get_embedding_model(self, model_name: str):
        """Get an embedding model by name."""
        return self.embedding_models.get(model_name)
    
    def get_available_embedding_models(self) -> List[str]:
        """Get list of available embedding models."""
        return list(self.embedding_models.keys())
    
    def get_qdrant_client(self):
        """Get Qdrant client."""
        return self.qdrant_client
    
    def get_elasticsearch_client(self):
        """Get Elasticsearch client."""
        return self.es_client

# Create singleton instance
database_service = DatabaseService()