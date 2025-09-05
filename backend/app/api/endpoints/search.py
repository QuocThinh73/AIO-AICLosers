import asyncio
from fastapi import APIRouter, UploadFile, File, Depends
from typing import Optional
from app.models.schemas import BaseSearchRequest, TemporalSearchRequest
from app.utils.rerank import rrf
from app.utils.translate import translate_text
from functools import lru_cache
from io import BytesIO
from PIL import Image


router = APIRouter()

@lru_cache()
def get_search_service():
   from app.services.search_service import SearchService
   return SearchService()

@router.post("/base_search")
async def base_search(base_search_request: BaseSearchRequest = Depends(BaseSearchRequest.as_form), embedding_image: Optional[UploadFile] = File(None), search_service = Depends(get_search_service)):
   # Convert string to list of batch ids
   include_batch_ids = [id.strip() for id in base_search_request.include_batch_ids.split(",")] if base_search_request.include_batch_ids else None
   exclude_batch_ids = [id.strip() for id in base_search_request.exclude_batch_ids.split(",")] if base_search_request.exclude_batch_ids else None

   # Translate text from Vietnamese to English if needed
   if base_search_request.use_translation:
      base_search_request.embedding_text = await translate_text(text=base_search_request.embedding_text, src_lang="vi", dest_lang="en") if base_search_request.embedding_text else base_search_request.embedding_text
      base_search_request.captioning_text = await translate_text(text=base_search_request.captioning_text, src_lang="vi", dest_lang="en") if base_search_request.captioning_text else base_search_request.captioning_text
      base_search_request.object_detection_text = await translate_text(text=base_search_request.object_detection_text, src_lang="vi", dest_lang="en") if base_search_request.object_detection_text else base_search_request.object_detection_text

   # Search
   results = []
   if base_search_request.embedding_text:
      embedding_text_result = search_service.search_openclip(text=base_search_request.embedding_text, image=None, top_k=base_search_request.top_k, include_batch_ids=include_batch_ids, exclude_batch_ids=exclude_batch_ids)
      results.append(embedding_text_result)
   if embedding_image:
      embedding_image = Image.open(BytesIO(await embedding_image.read())).convert("RGB")
      embedding_image_result = search_service.search_openclip(image=embedding_image, text=None, top_k=base_search_request.top_k, include_batch_ids=include_batch_ids, exclude_batch_ids=exclude_batch_ids)
      results.append(embedding_image_result)
   if base_search_request.captioning_text:
      captioning_result = search_service.search_caption(base_search_request.captioning_text, base_search_request.top_k)
      results.append(captioning_result)
   if base_search_request.ocr_text:
      ocr_result = search_service.search_ocr(base_search_request.ocr_text, base_search_request.top_k)
      results.append(ocr_result)
   if base_search_request.object_detection_text:
      object_detection_result = search_service.search_object(base_search_request.object_detection_text, base_search_request.top_k)
      results.append(object_detection_result)

   # Rerank
   results = rrf(results, base_search_request.top_k)

   # Return results
   return results

@router.post("/temporal_search")
def temporal_search(temporal_search_request: TemporalSearchRequest):
   return {"message": "Not implemented yet"}