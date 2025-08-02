from pydantic import BaseModel


class BaseSearchRequest(BaseModel):
    embedding: bool
    captioning: bool
    ocr: bool
    object_detection: bool
    embedding_text: str
    captioning_text: str
    ocr_text: str
    object_detection_text: str
    top_k: int

class TemporalSearchRequest(BaseModel):
    pass
