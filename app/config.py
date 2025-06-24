import torch
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
UPLOAD_FOLDER = 'app/static/images'
DATABASE_FOLDER = 'database'
MAPPING_JSON = os.path.join(DATABASE_FOLDER, "id2path.json")
EMBEDDING_MODELS = {
    "CLIP ViT-B/32": {
        "model_type": "clip",
        "backbone": "ViT-B/32",
        "faiss_database_name": "clip_faiss.bin",
        },
    "OpenCLIP ViT-B-32 laion2b_s34b_b79k": {
        "model_type": "openclip",
        "backbone": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
        "faiss_database_name": "openclip_faiss.bin"
        },
}
OBJECTS = ["car", "person", "dog", "cat", "bird", "fish", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "lion", "tiger", "monkey", "snake", "rabbit", "squirrel", "fox", "wolf", "deer"]





