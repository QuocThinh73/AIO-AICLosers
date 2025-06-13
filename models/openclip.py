import torch
import models.openclip as openclip
import numpy as np

from models.base_vlm import BaseVLM

class OpenCLIP(BaseVLM):
    def __init__(self, backbone, pretrained, device="cpu"):
        self.device = device
        self.model, _, self.preprocess = openclip.create_model_and_transforms(backbone, pretrained)
        self.model.eval()
        self.tokenizer = openclip.get_tokenizer(backbone)

    def encode_image(self, image):
        image = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().astype(np.float32).reshape(-1)
    
    def encode_text(self, text):
        text = self.tokenizer(text)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().astype(np.float32).reshape(-1)