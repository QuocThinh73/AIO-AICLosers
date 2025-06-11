from lavis.models import load_model_and_preprocess
import torch
import numpy as np
from PIL import Image

from models.base_vlm import BaseVLM

class BLIP2(BaseVLM):
    def __init__(self, model_type="pretrain", device="cpu"):
        self.device = device
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name="blip2_feature_extractor", model_type=model_type, is_eval=True, device=device
        )
        self.model.eval()

    def encode_image(self, image):
        image_tensor = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.extract_features({"image": image_tensor, "text_input": [""]}, mode="image")
        image_features = image_features.image_embeds[:, 0, :]
        return image_features.cpu().numpy().astype(np.float32).reshape(-1)

    def encode_text(self, text: str) -> np.ndarray:
        text_input = self.txt_processors["eval"](text).to(self.device)
        with torch.no_grad():
            text_features = self.model.extract_features({"image": None, "text_input": [text_input]}, mode="text")
        text_features = text_features.text_embeds[:, 0, :]
        return text_features.cpu().numpy().astype(np.float32).reshape(-1)
