from pydantic import BaseModel
from typing import List, Optional
from PIL import Image


class BaseSearchRequest(BaseModel):
    use_embedding_text: bool = False
    use_embedding_image: bool = False
    use_captioning: bool = False
    use_ocr: bool = False
    use_object_detection: bool = False
    use_translation: bool = False

    embedding_text: Optional[str] = None
    captioning_text: Optional[str] = None
    ocr_text: Optional[str] = None
    object_detection_text: Optional[str] = None

    top_k: int = 100
    

class TemporalSearchRequest(BaseModel):
    pass
