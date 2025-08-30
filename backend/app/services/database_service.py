import logging
import os
from typing import Dict, Any, Optional, List
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
import faiss
import sys
import torch

# Thêm đường dẫn gốc dự án vào sys.path để import database module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

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
        
        # Xử lý đường dẫn tương đối - tạo đường dẫn tuyệt đối từ đường dẫn tương đối
        # Nếu đường dẫn đã là tuyệt đối, giữ nguyên; nếu là tương đối, thêm thư mục gốc
        self.database_path = self._resolve_path(settings.DATABASE_FOLDER)
        self.embeddings_path = self._resolve_path(settings.EMBEDDING_FOLDER)
        self.mapping_json = self._resolve_path(settings.MAPPING_JSON)
        
        # Initialize connections
        self._init_embedding_models()
        self._connect_qdrant()
        self._connect_elasticsearch()
    
    def _init_embedding_models(self):
        """Initialize available embedding models using FAISS."""
        try:
            import faiss
            from database.my_faiss import Faiss
            import open_clip
        except ImportError as e:
            logging.error(f"FAISS, my_faiss, or open_clip module not available: {str(e)}. Embedding search will not work.")
            return
            
        for model_name, model_info in settings.EMBEDDING_MODELS.items():
            try:
                embeddings_file = os.path.join(self.embeddings_path, model_info.get("embeddings_file", ""))
                
                if os.path.exists(embeddings_file) and os.path.exists(self.mapping_json):
                    logging.info(f"Loading embedding model {model_name} from {embeddings_file}")
                    
                    # Tải mô hình OpenCLIP trước khi khởi tạo Faiss
                    model_type = model_info.get("model_type")
                    backbone = model_info.get("backbone")
                    pretrained = model_info.get("pretrained")
                    
                    if model_type == "openclip":
                        try:
                            # Tải mô hình OpenCLIP
                            device = torch.device(settings.DEVICE)
                            model, _, preprocess = open_clip.create_model_and_transforms(
                                backbone, 
                                pretrained=pretrained,
                                device=device
                            )
                            tokenizer = open_clip.get_tokenizer(backbone)
                            
                            # Tạo wrapper đơn giản để sử dụng với Faiss
                            class OpenCLIPWrapper:
                                def __init__(self, model, tokenizer, device):
                                    self.model = model
                                    self.tokenizer = tokenizer
                                    self.device = device
                                
                                def encode_text(self, text):
                                    with torch.no_grad():
                                        text_tokens = self.tokenizer([text]).to(self.device)
                                        text_features = self.model.encode_text(text_tokens)
                                        return text_features.cpu().numpy()
                            
                            # Khởi tạo wrapper
                            clip_model = OpenCLIPWrapper(model, tokenizer, device)
                            logging.info(f"Successfully loaded OpenCLIP model {backbone} with {pretrained}")
                        except Exception as clip_error:
                            logging.error(f"Error loading OpenCLIP model: {str(clip_error)}")
                            continue
                    else:
                        logging.error(f"Unsupported model type: {model_type}")
                        continue
                    
                    # Khởi tạo object Faiss với model đã tải
                    faiss_instance = Faiss(model=clip_model)
                    
                    # Load FAISS index và mapping
                    try:
                        # Load FAISS index từ file
                        faiss_instance.load_embeddings(embeddings_file)
                        
                        # Load mapping từ JSON
                        faiss_instance.load_mapping(self.mapping_json)
                        
                        # Lưu instance vào dictionary
                        self.embedding_models[model_name] = faiss_instance
                        logging.info(f"Successfully loaded embedding model {model_name} with index")
                    except Exception as load_error:
                        logging.error(f"Error loading index for {model_name}: {str(load_error)}")
                else:
                    logging.warning(f"Embedding file {embeddings_file} or mapping file {self.mapping_json} not found")
            except Exception as e:
                logging.error(f"Failed to initialize embedding model {model_name}: {str(e)}")
    
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

    def _resolve_path(self, path: str) -> str:
        """Convert relative path to absolute path if needed.
        
        If the path is already absolute, return it unchanged.
        If the path is relative, make it absolute based on project root.
        """
        if os.path.isabs(path):
            return path
            
        # Determine project root by going up one level from the backend folder
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
        
        # If path is relative to data folder (như ../data/embeddings)
        if path.startswith('../'):
            return os.path.normpath(os.path.join(project_root, path[3:]))
            
        # If path is just a filename or subfolder (như embeddings hoặc id2path.json)
        # Giả định rằng nó nằm trong thư mục data
        if settings.DATABASE_FOLDER and not os.path.isabs(settings.DATABASE_FOLDER):
            # Xử lý đường dẫn tương đối tới DATABASE_FOLDER
            if settings.DATABASE_FOLDER.startswith('../'):
                data_folder = os.path.normpath(os.path.join(project_root, settings.DATABASE_FOLDER[3:]))
            else:
                data_folder = os.path.normpath(os.path.join(project_root, settings.DATABASE_FOLDER))
            # Thêm path vào data_folder
            return os.path.normpath(os.path.join(data_folder, path))
        else:
            # Nếu không có DATABASE_FOLDER, giả định là thư mục data ở thư mục gốc
            return os.path.normpath(os.path.join(project_root, 'data', path))

# Create singleton instance
database_service = DatabaseService()