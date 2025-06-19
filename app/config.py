import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
UPLOAD_FOLDER = 'app/static/images'
DATA_FOLDER = 'data'
DATABASE_FOLDER = 'database'
EMBEDDING_MODELS = {
    "CLIP ViT-B/32": {
        "model_type": "clip",
        "backbone": "ViT-B/32",
        "faiss_database_name": "clip_faiss.bin",
        "id2path_name": "clip_id2path.pkl",
        },
    "OpenCLIP ViT-B-32 laion2b_s34b_b79k": {
        "model_type": "openclip",
        "backbone": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
        "faiss_database_name": "openclip_faiss.bin",
        "id2path_name": "openclip_id2path.pkl",
        },
}
OBJECTS = ["car", "person", "dog", "cat", "bird", "fish", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "lion", "tiger", "monkey", "snake", "rabbit", "squirrel", "fox", "wolf", "deer"]





