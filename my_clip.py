import clip
import torch

class MyCLIP(BaseVLM):
    def __init__(self, clip_backbone="ViT-B/32", device="cpu"):
        self.device = device
        self.model, self.processor = clip.load(clip_backbone, device=device)
        self.model.eval()

    def encode_text(self, text: str) -> np.ndarray:
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().astype(np.float32).reshape(-1)

    def encode_image(self, image):
        img_tensor = self.processor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_features = self.model.encode_image(img_tensor)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        return img_features.cpu().numpy().astype(np.float32).reshape(-1)
