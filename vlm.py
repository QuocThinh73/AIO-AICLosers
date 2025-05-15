from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import torch
from typing import Union, List

class BaseVLM(ABC):
    """Abstract base class for Vision-Language Models.
    
    All VLM implementations should inherit from this class and implement
    the required methods for encoding text and images.
    """
    
    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text into a feature vector.
        
        Args:
            text (str): The input text to encode
            
        Returns:
            np.ndarray: The normalized embedding vector
        """
        pass
    
    @abstractmethod
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode image into a feature vector.
        
        Args:
            image (PIL.Image): The input image to encode
            
        Returns:
            np.ndarray: The normalized embedding vector
        """
        pass
    
    def encode_batch_images(self, images: List[Image.Image]) -> np.ndarray:
        """Encode a batch of images into feature vectors.
        
        Default implementation processes images one by one.
        Subclasses can override for more efficient batch processing.
        
        Args:
            images (List[PIL.Image]): List of input images to encode
            
        Returns:
            np.ndarray: Array of normalized embedding vectors
        """
        embeddings = []
        for image in images:
            embeddings.append(self.encode_image(image))
        return np.array(embeddings)


class CLIPModel(BaseVLM):
    """Implementation of CLIP (Contrastive Language-Image Pre-Training) model."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """Initialize CLIP model.
        
        Args:
            model_name (str): The CLIP model variant to use
        """
        # Import here to avoid loading dependencies unless this class is used
        from transformers import CLIPProcessor, CLIPModel
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using CLIP text encoder.
        
        Args:
            text (str): The input text to encode
            
        Returns:
            np.ndarray: The normalized embedding vector
        """
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            
        # Normalize and convert to numpy
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features.cpu().numpy()[0]
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode image using CLIP image encoder.
        
        Args:
            image (PIL.Image): The input image to encode
            
        Returns:
            np.ndarray: The normalized embedding vector
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
        # Normalize and convert to numpy
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features.cpu().numpy()[0]
    
    def encode_batch_images(self, images: List[Image.Image]) -> np.ndarray:
        """Encode a batch of images using CLIP image encoder.
        
        Args:
            images (List[PIL.Image]): List of input images to encode
            
        Returns:
            np.ndarray: Array of normalized embedding vectors
        """
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
        # Normalize and convert to numpy
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features.cpu().numpy()


class BLIPModel(BaseVLM):
    """Implementation of BLIP (Bootstrapping Language-Image Pre-training) model."""
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        """Initialize BLIP model.
        
        Args:
            model_name (str): The BLIP model variant to use
        """
        # Import here to avoid loading dependencies unless this class is used
        from transformers import BlipProcessor, BlipModel
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BlipModel.from_pretrained(model_name).to(self.device)
        self.processor = BlipProcessor.from_pretrained(model_name)
        
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text using BLIP text encoder.
        
        Args:
            text (str): The input text to encode
            
        Returns:
            np.ndarray: The normalized embedding vector
        """
        inputs = self.processor(text=text, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            
        # Normalize and convert to numpy
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features.cpu().numpy()[0]
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode image using BLIP image encoder.
        
        Args:
            image (PIL.Image): The input image to encode
            
        Returns:
            np.ndarray: The normalized embedding vector
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
        # Normalize and convert to numpy
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features.cpu().numpy()[0] 