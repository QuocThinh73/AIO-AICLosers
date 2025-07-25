import os
from app.rerank import rrf
from app.config import CAPTIONS_COLLECTION

def perform_text_search(query, models, database, topK):
    """Perform text-based search using specified models."""
    if not query:
        return {}
    
    text_results = {}
    for model in models:
        faiss_handler = database.embedding_models[f'{model}']
        _, _, paths = faiss_handler.text_search(query=query, top_k=topK)
        text_results[f'{model}_text'] = paths
    
    return text_results


def perform_image_search(uploaded_image, models, database, topK):
    """Perform image-based search using specified models."""
    if not uploaded_image:
        return {}
    
    image_results = {}
    for model in models:
        faiss_handler = database.embedding_models[f'{model}']
        _, _, paths = faiss_handler.image_search(query_image=uploaded_image, top_k=topK)
        image_results[f'{model}_image'] = paths
    
    return image_results

def perform_caption_search(query, database, topK):
    """Perform caption-based search (disabled - no Qdrant)."""
    if not query:
        return {}
    
    # Caption search disabled without Qdrant
    print("[INFO] Caption search disabled (no Qdrant)")
    return {}

def perform_ocr_search(ocr_text, models, database, topK):
    """Perform OCR-based search (TODO: implement when available)."""
    # TODO: Implement OCR text search logic here
    return {}

def perform_grounding_search(query, database, topK):
    """Perform GroundingDINO-based object detection search."""
    if not query:
        return {}
    
    grounding_results = {}
    
    # Use GroundingDINO for object detection search
    paths = database.grounding_search(query, topK)
    grounding_results['grounding_dino'] = paths
    
    return grounding_results

def filter_by_objects(results, scores, objects):
    """Previously filtered search results by detected objects using YOLOv8.
    This functionality has been removed.
    
    Args:
        results (list): List of keyframe paths
        scores (list): List of corresponding relevance scores
        objects (list): List of object class names to filter by (no longer used)
        
    Returns:
        tuple: (results, scores) unchanged
    """
    # YOLOv8 filtering has been removed
    print(f"[INFO] Object filtering is disabled: returning {len(results)} results unchanged")
    return results if results else [], scores if scores else []

def perform_unified_search(uploaded_image, search_params, database):
    """Perform unified search combining all search types using RRF fusion.
    
    Search flow:
    1. Collect results from all search methods (text, image, grounding)
    2. Apply RRF fusion to rank results across all methods
    
    Args:
        uploaded_image: Optional image query
        search_params: Search parameters dictionary
        database: Database instance
        
    Returns:
        tuple: (paths, scores) - Ranked results
    """
    query = search_params['query']
    ocr_text = search_params['ocr_text']
    models = search_params['models']
    objects = search_params['objects']
    topK = search_params['topK']
    
    print(f"[INFO] Unified search with query: '{query}', models: {models}, objects: {objects}")
    
    # Collect results from different search types
    all_results = {}
    
    # 1. Text-based search using embedding models
    text_results = perform_text_search(query, models, database, topK)
    all_results.update(text_results)
    print(f"[INFO] Text search results: {len(text_results)} model sets")
    
    # 2. Image-based search using embedding models (if image provided)
    image_results = perform_image_search(uploaded_image, models, database, topK)
    all_results.update(image_results)
    print(f"[INFO] Image search results: {len(image_results)} model sets")
    
    # 3. GroundingDINO object detection search (participates in RRF ranking)
    if query:  # Only if text query is provided
        grounding_results = perform_grounding_search(query, database, topK)
        all_results.update(grounding_results)
        print(f"[INFO] GroundingDINO search results: {len(grounding_results)} sets")
    
    # Apply RRF (Reciprocal Rank Fusion) to combine results from all models
    # This includes BOTH embedding models AND GroundingDINO results
    if all_results:
        print(f"[INFO] Applying RRF fusion across {len(all_results)} result sets")
        paths, scores = rrf(all_results, k_rrf=100)  # Default k_rrf value
        print(f"[INFO] RRF fusion results: {len(paths)} paths")
        
        # Apply YOLOv8 object filtering if requested (filter only, not for ranking)
        if search_params.get('objects'):
            print(f"[INFO] Applying YOLOv8 filtering for objects: {search_params['objects']}")
            paths, scores = perform_object_filtering(search_params['objects'], database, paths, scores)
            print(f"[INFO] After YOLOv8 filtering: {len(paths)} results")
    else:
        print("[WARNING] No search results found from any method")
        paths, scores = [], []
        
    return paths, scores


def format_search_response(paths, scores, uploaded_image, search_params, database):
    """Format search results into API response.
    
    Process paths to ensure proper format for frontend display:
    - Converts absolute paths to relative paths that match Flask routes
    - Formats keyframe paths in the expected pattern: L01/V003/L01_V003_015190.jpg
    
    Args:
        paths (list): List of absolute paths to keyframes
        scores (list): Corresponding relevance scores
        uploaded_image: Optional image query
        search_params (dict): Search parameters
        database (Database): Database instance
        
    Returns:
        dict: Formatted search results for API response
    """
    formatted_paths = []
    formatted_filenames = []
    
    for path in paths:
        try:
            # Extract filename first
            filename = os.path.basename(path)
            
            # Parse filename to get pattern like L01_V003_015190.jpg
            if '_' in filename and (filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png')):
                parts = filename.split('_')
                if len(parts) >= 3:  # Has lesson and video components
                    lesson = parts[0]  # e.g., L01
                    video = parts[1]   # e.g., V003
                    
                    # Create correctly formatted path for Flask route
                    formatted_path = f"{lesson}/{video}/{filename}"
                    formatted_paths.append(formatted_path)
                    formatted_filenames.append(filename)
                    
                    print(f"[DEBUG] Formatted path: {formatted_path}")
                else:
                    # Fallback: just use filename directly
                    formatted_paths.append(filename)
                    formatted_filenames.append(filename)
                    print(f"[WARNING] Could not parse proper path structure from: {filename}")
            else:
                # Fallback: just use filename directly
                formatted_paths.append(filename)
                formatted_filenames.append(filename)
                print(f"[WARNING] Could not parse proper path structure from: {filename}")
                
        except Exception as e:
            print(f"[ERROR] Failed to format path {path}: {e}")
            # In case of error, just use the original path with database path stripped
            rel_path = path.replace(str(database.database_path), '', 1).strip('/\\')
            formatted_paths.append(rel_path)
            formatted_filenames.append(os.path.basename(path))
    
    return {
        'paths': formatted_paths,
        'scores': scores,
        'filenames': formatted_filenames,
        'search_info': {
            'text_query': bool(search_params['query']),
            'caption_query': bool(search_params['query']),
            'image_query': bool(uploaded_image),
            'ocr_query': bool(search_params['ocr_text']),
            'object_filters': len(search_params['objects']),
            'models_used': search_params['models']
        }
    }