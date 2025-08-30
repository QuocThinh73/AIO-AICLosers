from pydantic import BaseModel
from fastapi import Form
from typing import Optional


class BaseSearchRequest(BaseModel):
    use_translation: bool = False

    embedding_text: Optional[str] = None
    captioning_text: Optional[str] = None
    ocr_text: Optional[str] = None
    object_detection_text: Optional[str] = None

    include_batch_ids: Optional[str] = None
    exclude_batch_ids: Optional[str] = None

    top_k: int = 100
    
    @classmethod
    def as_form(
        cls,
        use_translation: bool = Form(False),

        embedding_text: Optional[str] = Form(None),
        captioning_text: Optional[str] = Form(None),
        ocr_text: Optional[str] = Form(None),
        object_detection_text: Optional[str] = Form(None),

        include_batch_ids: Optional[str] = Form(None),
        exclude_batch_ids: Optional[str] = Form(None),

        top_k: int = Form(100),
    ) -> "BaseSearchRequest":
        return cls(
            use_translation=use_translation,
            embedding_text=embedding_text,
            captioning_text=captioning_text,
            ocr_text=ocr_text,
            object_detection_text=object_detection_text,
            include_batch_ids=include_batch_ids,
            exclude_batch_ids=exclude_batch_ids,
            top_k=top_k,
        )

class TemporalSearchRequest(BaseModel):
    pass
