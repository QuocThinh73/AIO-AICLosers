from abc import ABC, abstractmethod

class BaseVLM(ABC):
    @abstractmethod
    def encode_text(self, text):
        pass
    
    @abstractmethod
    def encode_image(self, image):
        pass