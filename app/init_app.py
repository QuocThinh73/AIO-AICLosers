import os
from models.clip import CLIP
from models.openclip import OpenCLIP
from faiss_index import FaissIndex
from app.config import *

def load_database():
    database = {}
    
    database_path = os.path.abspath(DATABASE_FOLDER)
    
    for model_name, model_info in EMBEDDING_MODELS.items():
        if (model_info["model_type"] == "clip"):
            model = CLIP(clip_backbone=model_info["backbone"], device=DEVICE)
        elif (model_info["model_type"] == "openclip"):
            model = OpenCLIP(backbone=model_info["backbone"], pretrained=model_info["pretrained"], device=DEVICE)
        faiss = FaissIndex(model=model)
        faiss.load(os.path.join(database_path, model_info["faiss_database_name"]), os.path.join(database_path, model_info["id2path_name"]))
        database[model_name] = faiss
    
    return database