import os
import sys
import traceback
from faiss_index import Faiss
from app.config import *

# Import Elasticsearch directly to ensure it's available
try:
    from elasticsearch import Elasticsearch
    ELASTICSEARCH_AVAILABLE = True
    print("[INFO] Elasticsearch module found and imported successfully")
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    print("[WARNING] Elasticsearch module not found, some features will be disabled")

class Database:
    """
    Database class for managing paths, embedding models, and search functionality.
    
    This class initializes all necessary paths and loads pre-trained embedding models
    with their corresponding FAISS indices for efficient similarity search.
    """
    
    def __init__(self):
        """
        Initialize database with paths and load embedding models.
        """
        # Set up absolute paths for all database components
        self.database_path = os.path.abspath(DATABASE_FOLDER)
        self.keyframes_path = os.path.abspath(KEYFRAMES_FOLDER)
        self.shots_path = os.path.abspath(SHOTS_FOLDER)
        self.videos_path = os.path.abspath(VIDEOS_FOLDER)
        self.embeddings_path = os.path.abspath(EMBEDDING_FOLDER)
        self.mapping_json = os.path.abspath(MAPPING_JSON)
        
        # Load embedding models and available object classes
        self.embedding_models = self.load_embedding_models()
        self.objects = OBJECTS
        
        # Load GroundingDINO model for object detection search
        self.grounding_dino = self.load_grounding_dino()
        
        print(f"Database initialized successfully!")
        print(f"- Embedding models: {len(self.embedding_models)}")
        print(f"- GroundingDINO: {'OK' if self.grounding_dino and self.grounding_dino.model_loaded else 'FAIL'}")
        print(f"- Object classes: {len(self.objects)}")
        
    def load_embedding_models(self):
        """
        Load lightweight embedding model optimized for 6GB VRAM.
        Only loads the smallest available model to minimize memory usage.
        
        Returns:
            dict: Dictionary mapping model names to initialized FAISS handlers
        """
        embedding_models = {}
        
        print("[INFO] Loading lightweight embedding models...")
        
        try:
            for model_name, model_info in EMBEDDING_MODELS.items():
                print(f"[INFO] Loading {model_name} (lightest model)...")
                
                # Initialize OpenCLIP model
                if model_info["model_type"] == "openclip":
                    from models.openclip import OpenCLIP
                    model = OpenCLIP(
                        backbone=model_info["backbone"], 
                        pretrained=model_info["pretrained"], 
                        device=DEVICE
                    )
                    
                    # Create FAISS handler
                    faiss = Faiss(model=model)
                    
                    # Load pre-computed embeddings if available
                    embeddings_file = os.path.join(self.embeddings_path, model_info["embeddings_file"])
                    if os.path.exists(embeddings_file) and os.path.exists(self.mapping_json):
                        faiss.load(embeddings_file, self.mapping_json)
                        print(f"[OK] Loaded embeddings for {model_name}")
                    else:
                        print(f"[WARNING] No embeddings found for {model_name}")
                    
                    embedding_models[model_name] = faiss
                    
                # Clear memory after each model load
                import gc
                gc.collect()
                if DEVICE == 'cuda':
                    import torch
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            print(f"[ERROR] Failed to load embedding models: {e}")
            print("[INFO] Continuing with GroundingDINO only...")
            
        print(f"[INFO] Loaded {len(embedding_models)} embedding models")
        return embedding_models
    
    def load_grounding_dino(self):
        """
        Load GroundingDINO model for object detection search.
        
        Returns:
            GroundingDINO: Initialized GroundingDINO model instance
        """
        try:
            from models.groundingdino import GroundingDINO
            grounding_model = GroundingDINO(device=DEVICE)
            
            if grounding_model.model_loaded:
                print("[INFO] GroundingDINO model loaded successfully in Database")
            else:
                print("[WARNING] GroundingDINO model failed to load, object detection search will be disabled")
            
            return grounding_model
            
        except ImportError as e:
            print(f"[WARNING] Failed to import GroundingDINO: {e}")
            print("[INFO] Object detection search will be disabled")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to load GroundingDINO: {e}")
            return None
    
    def get_embedding_models(self):
        """Get list of available embedding models."""
        return list(self.embedding_models.keys())
    
    def grounding_search(self, query: str, topK: int = 100):
        """
        Search for keyframes containing specific objects using GroundingDINO detection results.
        Tries to use Elasticsearch first, falls back to direct GroundingDINO processing if needed.
        
        Args:
            query (str): Object description or class name to search for
            topK (int): Maximum number of results to return
            
        Returns:
            list: List of keyframe paths containing the specified object
        """
        # Use the global flag to avoid duplicate checks
        if not ELASTICSEARCH_AVAILABLE:
            print("[WARNING] Elasticsearch module not available, falling back to direct processing")
            return self._fallback_grounding_search(query, topK)
        
        # Try to use Elasticsearch for efficient search
        try:
            # Connect to Elasticsearch and verify connection
            es = Elasticsearch("http://localhost:9200", request_timeout=5)
            
            # Check if Elasticsearch is running
            if not es.ping():
                print("[WARNING] Elasticsearch server not running at http://localhost:9200")
                return self._fallback_grounding_search(query, topK)
                
            print("[INFO] Successfully connected to Elasticsearch")
            
            # Check if groundingdino index exists
            if not es.indices.exists(index=GROUNDINGDINO_INDEX):
                print(f"[WARNING] Index '{GROUNDINGDINO_INDEX}' does not exist in Elasticsearch")
                return self._fallback_grounding_search(query, topK)
                
            print(f"[INFO] Found '{GROUNDINGDINO_INDEX}' index, proceeding with search")
            
            # Create query for groundingdino index
            es_query = {
                "size": topK,
                "query": {
                    "bool": {
                        "should": [
                            # Match in caption
                            {
                                "match": {
                                    "caption": {
                                        "query": query,
                                        "boost": 1.0
                                    }
                                }
                            },
                            # Match in object names (nested)
                            {
                                "nested": {
                                    "path": "objects",
                                    "query": {
                                        "match": {
                                            "objects.object": {
                                                "query": query,
                                                "boost": 2.0  # Higher weight for object matches
                                            }
                                        }
                                    },
                                    "score_mode": "max"
                                }
                            },
                            # Match in prompts (nested)
                            {
                                "nested": {
                                    "path": "objects",
                                    "query": {
                                        "match": {
                                            "objects.prompt": {
                                                "query": query,
                                                "boost": 1.5  # Medium weight for prompt matches
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
            
            # Execute the search
            response = es.search(index=GROUNDINGDINO_INDEX, body=es_query)
            hits = response.get('hits', {}).get('hits', [])
            
            # Extract results from groundingdino index
            results = []
            for hit in hits:
                source = hit.get('_source', {})
                keyframe_path = source.get('keyframe_path', '')
                
                # Convert to full path format expected by the system
                if keyframe_path:
                    # Clean path: remove './database/keyframes/' if present
                    keyframe_path = keyframe_path.replace('./database/keyframes/', '')
                    # Convert to full path
                    full_path = os.path.join(self.keyframes_path, keyframe_path)
                    if os.path.exists(full_path):
                        results.append(full_path)
                    else:
                        print(f"[WARNING] Path not found: {full_path}")
            
            print(f"[INFO] Found {len(results)} results from GroundingDINO index")
            
            # If we found results, return them
            if results:
                return results[:topK]
                
            # If no results, try the YOLOv8 index if it exists
            if es.indices.exists(index=YOLOV8_INDEX):
                print(f"[INFO] No results from {GROUNDINGDINO_INDEX}, checking {YOLOV8_INDEX}...")
                # YOLOv8 query is simpler and focused on object class names
                yolo_query = {
                    "size": topK,
                    "query": {
                        "nested": {
                            "path": "objects",
                            "query": {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["objects.object"],
                                    "fuzziness": "AUTO"
                                }
                            }
                        }
                    }
                }
                
                response = es.search(index=YOLOV8_INDEX, body=yolo_query)
                hits = response.get('hits', {}).get('hits', [])
                
                # Extract results from YOLOv8 index
                results = []
                for hit in hits:
                    source = hit.get('_source', {})
                    keyframe_path = source.get('keyframe_path', '')
                    
                    # Convert to full path format expected by the system
                    if keyframe_path:
                        # Clean path: remove './database/keyframes/' if present
                        keyframe_path = keyframe_path.replace('./database/keyframes/', '')
                        # Convert to full path
                        full_path = os.path.join(self.keyframes_path, keyframe_path)
                        if os.path.exists(full_path):
                            results.append(full_path)
                
                print(f"[INFO] Found {len(results)} results from YOLOv8 index")
                if results:
                    return results[:topK]
            
            # If no results from either index, use fallback only if requested
            print("[INFO] No results found in Elasticsearch indices")
            
            # Ask user if they want to run the slower direct processing
            print("[WARNING] Fallback to direct GroundingDINO processing would be slow")
            print("[INFO] Returning empty results to avoid performance issues")
            return []
            
        except Exception as e:
            print(f"[WARNING] Elasticsearch search failed: {str(e)}")
            print("[DEBUG] Exception details:")
            traceback.print_exc()
            return []
    
    def _fallback_grounding_search(self, query: str, topK: int = 100):
        """
        Fallback method that would process all keyframes with GroundingDINO directly.
        This is DISABLED to prevent application hanging when Elasticsearch is unavailable.
        
        Args:
            query (str): The search query for object detection
            topK (int): Maximum number of results to return
            
        Returns:
            list: Empty list as direct processing is disabled for performance
        """
        print("[INFO] Direct GroundingDINO processing is disabled for performance reasons")
        print("[INFO] Please ensure Elasticsearch is running with the GroundingDINO index")
        print("[INFO] Or run index_groundingdino_to_elasticsearch.py to create the index")
        print("[INFO] Returning empty results")
        
        return []