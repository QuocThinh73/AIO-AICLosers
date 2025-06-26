import torch
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
UPLOAD_FOLDER = 'app/static/images'
DATABASE_FOLDER = 'database'
KEYFRAMES_FOLDER = os.path.join(DATABASE_FOLDER, "keyframes")
SHOTS_FOLDER = os.path.join(DATABASE_FOLDER, "shots")
VIDEOS_FOLDER = os.path.join(DATABASE_FOLDER, "videos")
EMBEDDING_FOLDER = os.path.join(DATABASE_FOLDER, "embeddings")
MAPPING_JSON = os.path.join(DATABASE_FOLDER, "id2path.json")
EMBEDDING_MODELS = {
    # "CLIP ViT-B/32": {
    #     "model_type": "clip",
    #     "backbone": "ViT-B/32",
    #     "faiss_database_name": "clip_faiss.bin",
    #     },
    
    "OpenCLIP_ViT-B-16-SigLIP-512_webli_embeddings": {
        "model_type": "openclip",
        "backbone": "ViT-B-16-SigLIP-512",
        "pretrained": "webli",
        "embeddings_file": "OpenCLIP_ViT-B-16-SigLIP-512_webli_embeddings.bin"
    },
    "OpenCLIP_ViT-L-16-SigLIP-256_webli_embeddings": {
        "model_type": "openclip",
        "backbone": "ViT-L-16-SigLIP-256",
        "pretrained": "webli",
        "embeddings_file": "OpenCLIP_ViT-L-16-SigLIP-256_webli_embeddings.bin"
    },
}
OBJECTS = ["car", "person", "dog", "cat", "bird", "fish", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "lion", "tiger", "monkey", "snake", "rabbit", "squirrel", "fox", "wolf", "deer"]





