from fastapi import APIRouter
from backend.app.models.schemas import BaseSearchRequest, TemporalSearchRequest
from backend.app.services.search_service import search_service
router = APIRouter()

@router.post("/base_search")
def base_search(base_search_request: BaseSearchRequest):
    result = {
        "embedding": [],
        "captioning": [],
        "ocr": [],
        "object_detection": []
    }

    if base_search_request.embedding:
       embedding_result = search_service.embedding_search(base_search_request.embedding_text, base_search_request.top_k) 
       result["embedding"] = embedding_result
    if base_search_request.captioning:
       captioning_result = search_service.captioning_search(base_search_request.captioning_text, base_search_request.top_k)
       result["captioning"] = captioning_result
    if base_search_request.ocr:
       ocr_result = search_service.ocr_search(base_search_request.ocr_text, base_search_request.top_k)
       result["ocr"] = ocr_result
    if base_search_request.object_detection:
       object_detection_result = search_service.object_detection_search(base_search_request.object_detection_text, base_search_request.top_k)
       result["object_detection"] = object_detection_result
    return result

@router.post("/temporal_search")
def temporal_search(temporal_search_request: TemporalSearchRequest):
    pass