import torch
import os

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Application folders
UPLOAD_FOLDER = 'app/static/images'
DATABASE_FOLDER = 'database'

# Database subfolders
KEYFRAMES_FOLDER = os.path.join(DATABASE_FOLDER, "keyframes")
SHOTS_FOLDER = os.path.join(DATABASE_FOLDER, "shots")
VIDEOS_FOLDER = os.path.join(DATABASE_FOLDER, "videos")
EMBEDDING_FOLDER = os.path.join(DATABASE_FOLDER, "embeddings")

# Database files
MAPPING_JSON = os.path.join(DATABASE_FOLDER, "id2path.json")

# Available embedding models configuration
EMBEDDING_MODELS = {
    "OpenCLIP ViT-B-16-SigLIP-512 webli": {
        "model_type": "openclip",
        "backbone": "ViT-B-16-SigLIP-512",
        "pretrained": "webli",
        "embeddings_file": "OpenCLIP_ViT-B-16-SigLIP-512_webli_embeddings.bin"
    },
    "OpenCLIP ViT-L-16-SigLIP-256 webli": {
        "model_type": "openclip",
        "backbone": "ViT-L-16-SigLIP-256",
        "pretrained": "webli",
        "embeddings_file": "OpenCLIP_ViT-L-16-SigLIP-256_webli_embeddings.bin"
    },
}

# Available object classes for filtering
OBJECTS = ["car", "person", "dog", "cat", "bird", "fish", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "lion", "tiger", "monkey", "snake", "rabbit", "squirrel", "fox", "wolf", "deer"]





