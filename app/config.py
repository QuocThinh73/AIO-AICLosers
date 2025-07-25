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

# Available embedding models configuration (optimized for 6GB VRAM)
# Using only the lightest model: OpenCLIP_ViT-B-16_dfn2b (14.3MB)
EMBEDDING_MODELS = {
    "OpenCLIP ViT-B-16 dfn2b": {
        "model_type": "openclip",
        "backbone": "ViT-B-16",
        "pretrained": "dfn2b",
        "embeddings_file": "OpenCLIP_ViT-B-16_dfn2b_embeddings.bin"
    },
}

# Available search models (optimized for 6GB VRAM)
SEARCH_MODELS = {
    "OpenCLIP ViT-B-16 dfn2b": {
        "type": "embedding",
        "display_name": "OpenCLIP ViT-B-16 (Lightweight Semantic Search)"
    },
    "GroundingDINO": {
        "type": "detection",
        "display_name": "GroundingDINO (Object Detection)"
    }
}

# Available object classes for filtering - using full YOLO COCO 80 classes
OBJECTS = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Note: YOLOv8 functionality has been removed

# Qdrant collection names
CAPTIONS_COLLECTION = "captions"

# Elasticsearch configuration for object detection indices
ELASTICSEARCH_HOST = "http://localhost:9200"
GROUNDINGDINO_INDEX = "groundingdino"

# GroundingDINO configuration
GROUNDINGDINO_ENABLED = True
GROUNDINGDINO_BOX_THRESHOLD = 0.35
GROUNDINGDINO_TEXT_THRESHOLD = 0.25

# Note: YOLOv8 filtering functionality has been removed




