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
    include_video_ids: Optional[str] = None
    exclude_video_ids: Optional[str] = None

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
        include_video_ids: Optional[str] = Form(None),
        exclude_video_ids: Optional[str] = Form(None),

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
            include_video_ids=include_video_ids,
            exclude_video_ids=exclude_video_ids,
            top_k=top_k,
        )


class TemporalEvent(BaseModel):
    """Đại diện cho một sự kiện cần tìm kiếm theo thứ tự thời gian"""
    query: str  # Truy vấn tìm kiếm (text)
    event_id: str  # ID để xác định sự kiện (E1, E2...)

class TemporalSearchRequest(BaseModel):
    """Request model cho tìm kiếm theo thứ tự thời gian"""
    events: list[TemporalEvent]  # Danh sách các sự kiện cần tìm theo thứ tự
    top_k: int = 100  # Số kết quả tối đa trả về cho mỗi event
    use_translation: bool = False  # Có dịch query hay không
    
    # Các tham số lọc (tương tự BaseSearchRequest)
    include_batch_ids: Optional[str] = None
    exclude_batch_ids: Optional[str] = None
    include_video_ids: Optional[str] = None
    exclude_video_ids: Optional[str] = None
