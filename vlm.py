from abc import ABC, abstractmethod
import numpy as np
import torch

class BaseVLM(ABC):
    @abstractmethod
    def encode_text(self, text):
        pass
    
    @abstractmethod
    def encode_image(self, image):
        pass