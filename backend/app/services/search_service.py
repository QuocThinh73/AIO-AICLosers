import os
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from io import BytesIO
from PIL import Image
import numpy as np
from backend.app.core.config import (
    DATABASE_FOLDER, KEYFRAMES_FOLDER, SHOTS_FOLDER, VIDEOS_FOLDER,
    OCR_INDEX, OBJECT_DETECTION_INDEX, settings
)

# Import database service
from backend.app.services.database_service import database_service

class SearchService:
    """Service for handling various types of search operations."""
    
    def __init__(self):
        """Initialize search service with necessary connections."""
        # Set up absolute paths for all database components
        self.database_path = os.path.abspath(DATABASE_FOLDER) if DATABASE_FOLDER else ""
        self.keyframes_path = os.path.abspath(KEYFRAMES_FOLDER) if KEYFRAMES_FOLDER else ""
        self.shots_path = os.path.abspath(SHOTS_FOLDER) if SHOTS_FOLDER else ""
        self.videos_path = os.path.abspath(VIDEOS_FOLDER) if VIDEOS_FOLDER else ""
    
    
    def embedding_search(self, embedding_text: str, top_k: int, embedding_models: List[str] = None) -> List[Dict[str, Any]]:
        """Perform text-to-image search using embedding models."""
        if not embedding_text:
            return []
            
        # Get available models from database service
        available_models = database_service.get_available_embedding_models()
        
        # If specific models requested, filter to only those that are available
        if embedding_models and isinstance(embedding_models, list):
            models_to_use = [model for model in embedding_models if model in available_models]
        else:
            models_to_use = available_models
            
        if not models_to_use:
            return []
        
        # Collect results from all requested models
        results = []
        for model_name in models_to_use:
            try:
                faiss_handler = database_service.get_embedding_model(model_name)
                if not faiss_handler:
                    continue
                    
                distances, indices, paths = faiss_handler.text_search(query=embedding_text, top_k=top_k)
                
                # Format results
                model_results = [{
                    'path': self._format_path(path),
                    'score': float(1.0 - distance),  # Convert distance to similarity score
                    'filename': os.path.basename(path),
                    'source': f'embedding_{model_name}'
                } for path, distance in zip(paths, distances)]
                
                results.extend(model_results)
            except Exception as e:
                logging.error(f"Error in embedding search with {model_name}: {str(e)}")
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def captioning_search(self, captioning_text: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform search using image captions in Qdrant."""
        if not captioning_text:
            return []
        
        # Get Qdrant client from database service
        qdrant_client = database_service.get_qdrant_client()
        if not qdrant_client:
            return []
        
        try:
            # Search in Qdrant collection
            search_results = qdrant_client.search(
                collection_name=settings.CAPTIONS_COLLECTION,
                query_vector=self._encode_text_for_qdrant(captioning_text),
                limit=top_k
            )
            
            # Format results
            results = [{
                'path': self._format_path(hit.payload.get('image_path', '')),
                'score': float(hit.score),
                'filename': os.path.basename(hit.payload.get('image_path', '')),
                'caption': hit.payload.get('caption', ''),
                'source': 'caption'
            } for hit in search_results]
            
            return results
        except Exception as e:
            logging.error(f"Error in caption search: {str(e)}")
            return []
    
    def _encode_text_for_qdrant(self, text: str) -> List[float]:
        """Encode text for Qdrant search - placeholder for actual encoding."""
        # This would normally use a text encoder compatible with your Qdrant setup
        # For now, returning a placeholder vector of appropriate dimension
        # You'll need to implement actual text encoding based on your model
        return [0.0] * 512  # Assuming 512 dimensions, adjust as needed
    
    def ocr_search(self, ocr_text: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform search using OCR text in Elasticsearch."""
        if not ocr_text:
            return []
        
        # Get Elasticsearch client from database service
        es_client = database_service.get_elasticsearch_client()
        if not es_client:
            return []
        
        try:
            # Search in Elasticsearch OCR index
            es_query = {
                "query": {
                    "match": {
                        "text": ocr_text
                    }
                },
                "size": top_k
            }
            
            response = es_client.search(
                index=OCR_INDEX,
                body=es_query
            )
            
            # Format results
            results = [{
                'path': self._format_path(hit['_source'].get('image_path', '')),
                'score': float(hit['_score']),
                'filename': os.path.basename(hit['_source'].get('image_path', '')),
                'text': hit['_source'].get('text', ''),
                'source': 'ocr'
            } for hit in response['hits']['hits']]
            
            return results
        except Exception as e:
            logging.error(f"Error in OCR search: {str(e)}")
            return []
    
    def object_detection_search(self, object_detection_text: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform search for objects detected in images using Elasticsearch."""
        if not object_detection_text:
            return []
        
        # Lấy ES client từ database service
        es_client = database_service.get_elasticsearch_client()
        if not es_client:
            return []
        
        try:
            # Tạo Elasticsearch query
            es_query = {
                "size": top_k,
                "query": {
                    "bool": {
                        "should": [
                            # Tìm trong caption
                            {
                                "match": {
                                    "caption": {
                                        "query": object_detection_text,
                                        "boost": 1.0
                                    }
                                }
                            },
                            # Tìm trong tên object
                            {
                                "nested": {
                                    "path": "objects",
                                    "query": {
                                        "match": {
                                            "objects.object": {
                                                "query": object_detection_text,
                                                "boost": 2.0  # Trọng số cao nhất cho object match
                                            }
                                        }
                                    },
                                    "score_mode": "max"
                                }
                            },
                            # Tìm trong prompts
                            {
                                "nested": {
                                    "path": "objects",
                                    "query": {
                                        "match": {
                                            "objects.prompt": {
                                                "query": object_detection_text,
                                                "boost": 1.5  # Trọng số trung bình cho prompt match
                                            }
                                        }
                                    },
                                    "score_mode": "max"
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                }
            }
            
            # Thực hiện tìm kiếm
            response = es_client.search(
                index=OBJECT_DETECTION_INDEX,
                body=es_query
            )
            
            # Format kết quả trả về
            results = []
            for hit in response['hits']['hits']:
                source = hit.get('_source', {})
                
                # Lọc objects liên quan đến query
                relevant_objects = []
                for obj in source.get('objects', []):
                    query_lower = object_detection_text.lower()
                    if (query_lower in obj.get('object', '').lower() or 
                        query_lower in obj.get('prompt', '').lower()):
                        relevant_objects.append(obj)
                
                # Tạo kết quả
                result_item = {
                    'path': self._format_path(source.get('keyframe_path', '')),
                    'score': float(hit['_score']),
                    'filename': os.path.basename(source.get('keyframe_path', '')),
                    'objects': relevant_objects if relevant_objects else source.get('objects', []),
                    'source': 'object_detection'
                }
                results.append(result_item)
            
            return results
        except Exception as e:
            logging.error(f"Error in object detection search: {str(e)}")
            return []
    
    def _format_path(self, path: str) -> str:
        """Format path for consistent response format."""
        # Remove database path prefix if present
        if self.database_path and path.startswith(self.database_path):
            return path.replace(self.database_path, '', 1)
        return path

# Initialize singleton instance
search_service = SearchService()