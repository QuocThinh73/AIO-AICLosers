from pydantic import BaseModel
from fastapi import Form
from typing import Optional


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

    include_batch_ids: Optional[str] = None
    exclude_batch_ids: Optional[str] = None

    top_k: int = 100
    
    @classmethod
    def as_form(
        cls,
        use_embedding_text: bool = Form(False),
        use_embedding_image: bool = Form(False),
        use_captioning: bool = Form(False),
        use_ocr: bool = Form(False),
        use_object_detection: bool = Form(False),
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
            use_embedding_text=use_embedding_text,
            use_embedding_image=use_embedding_image,
            use_captioning=use_captioning,
            use_ocr=use_ocr,
            use_object_detection=use_object_detection,
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
