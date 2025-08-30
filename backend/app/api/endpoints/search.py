import asyncio
from fastapi import APIRouter, UploadFile, File, Depends
from typing import Optional
from backend.app.models.schemas import BaseSearchRequest, TemporalSearchRequest
from backend.app.services.search_service import SearchService
from backend.app.utils.rerank import rrf
from backend.app.utils.translate import translate_text


router = APIRouter()


@router.post("/base_search")
def base_search(base_search_request: BaseSearchRequest = Depends(BaseSearchRequest.as_form), embedding_image: Optional[UploadFile] = File(None)):
   # Debug
   print(base_search_request.top_k, type(base_search_request.top_k))
   print(base_search_request.use_embedding_text, type(base_search_request.use_embedding_text))
   print(base_search_request.use_embedding_image, type(base_search_request.use_embedding_image))
   print(base_search_request.use_captioning, type(base_search_request.use_captioning))
   print(base_search_request.use_ocr, type(base_search_request.use_ocr))
   print(base_search_request.use_object_detection, type(base_search_request.use_object_detection))
   print(base_search_request.use_translation, type(base_search_request.use_translation))
   print(base_search_request.embedding_text, type(base_search_request.embedding_text))
   print(base_search_request.captioning_text, type(base_search_request.captioning_text))
   print(base_search_request.ocr_text, type(base_search_request.ocr_text))
   print(base_search_request.object_detection_text, type(base_search_request.object_detection_text))
   print(base_search_request.include_batch_ids, type(base_search_request.include_batch_ids))
   print(base_search_request.exclude_batch_ids, type(base_search_request.exclude_batch_ids))
   
   # Convert string to list of batch ids
   include_batch_ids = [id.strip() for id in base_search_request.include_batch_ids.split(",")] if base_search_request.include_batch_ids else None
   exclude_batch_ids = [id.strip() for id in base_search_request.exclude_batch_ids.split(",")] if base_search_request.exclude_batch_ids else None
   
   # Initialize search service
   search_service = SearchService()

   # Translate text from Vietnamese to English if needed
   if base_search_request.use_translation:
      base_search_request.embedding_text = asyncio.run(translate_text(text=base_search_request.embedding_text, src_lang="vi", dest_lang="en")) if base_search_request.use_embedding_text else base_search_request.embedding_text
      base_search_request.captioning_text = asyncio.run(translate_text(text=base_search_request.captioning_text, src_lang="vi", dest_lang="en")) if base_search_request.use_captioning else base_search_request.captioning_text
      base_search_request.object_detection_text = asyncio.run(translate_text(text=base_search_request.object_detection_text, src_lang="vi", dest_lang="en")) if base_search_request.use_object_detection else base_search_request.object_detection_text

   # Search
   results = []
   if base_search_request.use_embedding_text:
      embedding_text_result = search_service.search_openclip(text=base_search_request.embedding_text, image=None, top_k=base_search_request.top_k, include_batch_ids=include_batch_ids, exclude_batch_ids=exclude_batch_ids)
      results.append(embedding_text_result)
   if base_search_request.use_embedding_image:
      embedding_image_result = search_service.search_openclip(image=embedding_image, text=None, top_k=base_search_request.top_k, include_batch_ids=include_batch_ids, exclude_batch_ids=exclude_batch_ids)
      results.append(embedding_image_result)
   if base_search_request.use_captioning:
      captioning_result = search_service.search_caption(base_search_request.captioning_text, base_search_request.top_k)
      results.append(captioning_result)
   if base_search_request.use_ocr:
      ocr_result = search_service.search_ocr(base_search_request.ocr_text, base_search_request.top_k)
      results.append(ocr_result)
   if base_search_request.use_object_detection:
      object_detection_result = search_service.search_object(base_search_request.object_detection_text, base_search_request.top_k)
      results.append(object_detection_result)

   # Rerank
   results = rrf(results, base_search_request.top_k)

   # Return results
   return results

@router.post("/temporal_search")
def temporal_search(temporal_search_request: TemporalSearchRequest):
   return {"message": "Not implemented yet"}