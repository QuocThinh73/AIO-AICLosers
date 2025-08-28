from backend.app.database.qdrant import Qdrant
from backend.app.database.elasticsearch import Elasticsearch
from backend.app.core.config import get_settings
from PIL import Image

class DatabaseService:
    def __init__(self):
        self.qdrant = Qdrant(host=get_settings().QDRANT_HOST, port=get_settings().QDRANT_PORT)
        self.elasticsearch = Elasticsearch(host=get_settings().ELASTICSEARCH_HOST, port=get_settings().ELASTICSEARCH_PORT)

    def search_caption(self, text: str, top_k: int):
        return self.qdrant.search_caption(text=text, collection_name=get_settings().CAPTION_COLLECTION_NAME, limit=top_k, prefetch_limit=top_k*3)

    def search_openclip(self, text: str, image: Image, top_k: int):
        if text:
            return self.qdrant.search_openclip(text=text, image=None, collection_name=get_settings().OPENCLIP_COLLECTION_NAME, limit=top_k)
        elif image:
            return self.qdrant.search_openclip(text=None, image=image, collection_name=get_settings().OPENCLIP_COLLECTION_NAME, limit=top_k)