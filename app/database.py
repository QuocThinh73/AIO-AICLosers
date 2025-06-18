import os
from models.clip import CLIP
from models.openclip import OpenCLIP
from faiss_index import FaissIndex
from app.config import *

class Database:
    def __init__(self):
        self.database_path = os.path.abspath(DATABASE_FOLDER)
        self.embedding_models = self.load_embedding_models()
        
    def load_embedding_models(self):
        embedding_models = {}
        for model_name, model_info in EMBEDDING_MODELS.items():
            if (model_info["model_type"] == "clip"):
                model = CLIP(clip_backbone=model_info["backbone"], device=DEVICE)
            elif (model_info["model_type"] == "openclip"):
                model = OpenCLIP(backbone=model_info["backbone"], pretrained=model_info["pretrained"], device=DEVICE)
            faiss = FaissIndex(model=model)
            faiss.load(os.path.join(self.database_path, model_info["faiss_database_name"]), os.path.join(self.database_path, model_info["id2path_name"]))
            embedding_models[model_name] = faiss
            
        return embedding_models