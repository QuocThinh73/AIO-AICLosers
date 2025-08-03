from pydantic import BaseModel
from typing import List

class BaseSearchRequest(BaseModel):
    embedding: bool
    captioning: bool
    ocr: bool
    object_detection: bool
    translation: bool
    embedding_text: str
    captioning_text: str
    ocr_text: str
    object_detection_text: str
    top_k: int
    embedding_model: List[str]
    

class TemporalSearchRequest(BaseModel):
    pass
