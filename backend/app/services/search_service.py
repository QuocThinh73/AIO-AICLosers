from app.database.qdrant import Qdrant
from app.database.elasticsearch import Elasticsearch
from app.core.config import get_settings
from PIL import Image
from functools import lru_cache


class SearchService:
    def __init__(self):
        self.qdrant = Qdrant(host=get_settings().QDRANT_HOST,
                             port=get_settings().QDRANT_PORT)

    def search_caption(self, text: str, top_k: int, include_batch_ids=None, exclude_batch_ids=None, include_video_ids=None, exclude_video_ids=None):
        return self.qdrant.search_caption(search_query=text, collection_name=get_settings().COLLECTION_NAME, limit=top_k, include_video_ids=include_video_ids, exclude_video_ids=exclude_video_ids)

    def search_openclip(self, text: str, image: Image, top_k: int, include_batch_ids=None, exclude_batch_ids=None, include_video_ids=None, exclude_video_ids=None):
        if text:
            return self.qdrant.search_openclip(text=text, image=None, collection_name=get_settings().COLLECTION_NAME, limit=top_k, include_batch_ids=include_batch_ids, exclude_batch_ids=exclude_batch_ids, include_video_ids=include_video_ids, exclude_video_ids=exclude_video_ids)
        elif image:
            return self.qdrant.search_openclip(text=None, image=image, collection_name=get_settings().COLLECTION_NAME, limit=top_k, include_batch_ids=include_batch_ids, exclude_batch_ids=exclude_batch_ids, include_video_ids=include_video_ids, exclude_video_ids=exclude_video_ids)

    def search_ocr(self, ocr_text: str, top_k: int, include_batch_ids=None, exclude_batch_ids=None, include_video_ids=None, exclude_video_ids=None):
        pass

    def search_object(self, text: str, top_k: int, include_batch_ids=None, exclude_batch_ids=None, include_video_ids=None, exclude_video_ids=None):
        pass


@lru_cache()
def get_search_service():
    from app.services.search_service import SearchService
    return SearchService()
