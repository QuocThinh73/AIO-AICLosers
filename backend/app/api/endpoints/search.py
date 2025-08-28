from fastapi import APIRouter, UploadFile, File, Depends
from typing import Optional
from backend.app.models.schemas import BaseSearchRequest, TemporalSearchRequest
from backend.app.services.search_service import SearchService
from backend.app.utils.rerank import rrf


router = APIRouter()


@router.post("/base_search")
def base_search(base_search_request: BaseSearchRequest = Depends(), embedding_image: Optional[UploadFile] = File(None)):
   search_service = SearchService()

   if base_search_request.use_translation:
      base_search_request.embedding_text = search_service.translate_text(base_search_request.embedding_text) if base_search_request.use_embedding_text else base_search_request.embedding_text
      base_search_request.captioning_text = search_service.translate_text(base_search_request.captioning_text) if base_search_request.use_captioning else base_search_request.captioning_text
      base_search_request.ocr_text = search_service.translate_text(base_search_request.ocr_text) if base_search_request.use_ocr else base_search_request.ocr_text
      base_search_request.object_detection_text = search_service.translate_text(base_search_request.object_detection_text) if base_search_request.use_object_detection else base_search_request.object_detection_text

   results = []
   if base_search_request.use_embedding_text:
      embedding_result = search_service.search_openclip(text=base_search_request.embedding_text, top_k=base_search_request.top_k)
      results.append(embedding_result)
   if base_search_request.use_embedding_image:
      embedding_result = search_service.search_openclip(image=embedding_image, top_k=base_search_request.top_k)
      results.append(embedding_result)
   if base_search_request.use_captioning:
      captioning_result = search_service.search_caption(base_search_request.captioning_text, base_search_request.top_k)
      results.append(captioning_result)
   if base_search_request.use_ocr:
      ocr_result = search_service.search_ocr(base_search_request.ocr_text, base_search_request.top_k)
      results.append(ocr_result)
   if base_search_request.use_object_detection:
      object_detection_result = search_service.search_object(base_search_request.object_detection_text, base_search_request.top_k)
      results.append(object_detection_result)

   results = rrf(results, base_search_request.top_k)

   return results

@router.post("/temporal_search")
def temporal_search(temporal_search_request: TemporalSearchRequest):
   return {"message": "Not implemented yet"}