import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
UPLOAD_FOLDER = 'app/static/images'
DATA_FOLDER = 'data'
DATABASE_FOLDER = 'database'
EMBEDDING_MODELS = ["clip", "openclip"]





