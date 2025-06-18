import sys
import io
import os
from models.clip import CLIP
from models.openclip import OpenCLIP
from faiss_index import FaissIndex
from app.config import *

def load_database():
    database = {}
    
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    
    database_path = os.path.abspath(DATABASE_FOLDER)
    
    for model_name, model_info in EMBEDDING_MODELS.items():
        if (model_info["model_type"] == "clip"):
            model = CLIP(clip_backbone=model_info["backbone"], device=DEVICE)
        elif (model_info["model_type"] == "openclip"):
            model = OpenCLIP(backbone=model_info["backbone"], pretrained=model_info["pretrained"], device=DEVICE)
        faiss = FaissIndex(model=model)
        faiss.load(os.path.join(database_path, model_info["faiss_database_name"]), os.path.join(database_path, model_info["id2path_name"]))
        database[model_name] = faiss
            
    
    # # Khởi tạo các models
    # try:
    #     try:
    #         clip_model = CLIP(device=DEVICE)
    #         clip_faiss = FaissIndex(model=clip_model)
    #         clip_faiss.load(os.path.join(database_path, 'clip_faiss.bin'), os.path.join(database_path, 'clip_id2path.pkl'))
    #         database['clip_faiss'] = clip_faiss
    #     except Exception as e:
    #         pass
            
    #     try:
    #         openclip_model = OpenCLIP(backbone='ViT-B-32', pretrained='laion2b_s34b_b79k', device=DEVICE)
    #         openclip_faiss = FaissIndex(model=openclip_model)
    #         openclip_faiss.load(os.path.join(database_path, 'openclip_faiss.bin'), os.path.join(database_path, 'openclip_id2path.pkl'))
    #         database['openclip_faiss'] = openclip_faiss
    #     except Exception as e:
    #         pass
    
    # except Exception as e:
    #     import traceback
    #     print(f"Lỗi khi khởi tạo database: {str(e)}\n{traceback.format_exc()}")
    
    return database