from transformers import AutoProcessor, AutoModel, AutoTokenizer
import torch
import numpy as np

from models.base_vlm import BaseVLM

class SigLIP2(BaseVLM):
    def __init__(self, model_id="google/siglip2-base-patch16-224", device="cpu"):
        self.device = device
        self.model = AutoModel.from_pretrained(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model.eval()

    def encode_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().astype(np.float32).reshape(-1)
    
    def encode_text(self, text):
        inputs = self.tokenizer([text], padding="max_length", return_tensors="pt")
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().astype(np.float32).reshape(-1)