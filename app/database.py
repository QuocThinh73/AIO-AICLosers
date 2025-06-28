import os
from models.clip import CLIP
from models.openclip import OpenCLIP
from faiss_index import Faiss
from app.config import *

class Database:
    """
    Database class for managing paths, embedding models, and search functionality.
    
    This class initializes all necessary paths and loads pre-trained embedding models
    with their corresponding FAISS indices for efficient similarity search.
    """
    
    def __init__(self):
        """
        Initialize database with paths and load embedding models.
        """
        # Set up absolute paths for all database components
        self.database_path = os.path.abspath(DATABASE_FOLDER)
        self.keyframes_path = os.path.abspath(KEYFRAMES_FOLDER)
        self.shots_path = os.path.abspath(SHOTS_FOLDER)
        self.videos_path = os.path.abspath(VIDEOS_FOLDER)
        self.embeddings_path = os.path.abspath(EMBEDDING_FOLDER)
        self.mapping_json = os.path.abspath(MAPPING_JSON)
        
        # Load embedding models and available object classes
        self.embedding_models = self.load_embedding_models()
        self.objects = OBJECTS
        
    def load_embedding_models(self):
        """
        Load and initialize all configured embedding models with FAISS indices.
        
        Returns:
            dict: Dictionary mapping model names to initialized FAISS handlers
        """
        embedding_models = {}
        
        for model_name, model_info in EMBEDDING_MODELS.items():
            # Initialize appropriate model based on type
            if (model_info["model_type"] == "clip"):
                model = CLIP(clip_backbone=model_info["backbone"], device=DEVICE)
            elif (model_info["model_type"] == "openclip"):
                model = OpenCLIP(backbone=model_info["backbone"], pretrained=model_info["pretrained"], device=DEVICE)
            
            # Create FAISS handler and load pre-computed embeddings
            faiss = Faiss(model=model)
            faiss.load(os.path.join(self.embeddings_path, model_info["embeddings_file"]), self.mapping_json)
            embedding_models[model_name] = faiss
            
        return embedding_models