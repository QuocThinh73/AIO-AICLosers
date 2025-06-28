import os
from app.rerank import rrf


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


def perform_ocr_search(ocr_text, models, database, topK):
    """Perform OCR-based search (TODO: implement when available)."""
    # TODO: Implement OCR text search logic here
    return {}


def perform_object_filtering(objects, database):
    """Perform object-based filtering (TODO: implement when available)."""
    # TODO: Implement object filtering logic here
    return {}


def perform_unified_search(uploaded_image, search_params, database):
    """Perform unified search combining all search types."""
    query = search_params['query']
    ocr_text = search_params['ocr_text']
    models = search_params['models']
    objects = search_params['objects']
    topK = search_params['topK']
    
    # Collect results from different search types
    all_search_results = {}
    
    # 1. Text-based search
    text_results = perform_text_search(query, models, database, topK)
    all_search_results.update(text_results)
    
    # 2. Image-based search
    image_results = perform_image_search(uploaded_image, models, database, topK)
    all_search_results.update(image_results)
    
    # 3. OCR-based search
    ocr_results = perform_ocr_search(ocr_text, models, database, topK)
    all_search_results.update(ocr_results)
    
    # 4. Object filtering
    object_results = perform_object_filtering(objects, database)
    all_search_results.update(object_results)
    
    # Fuse all results using Reciprocal Rank Fusion
    paths, scores = rrf(all_search_results, k_rrf=60)
        
    return paths, scores


def format_search_response(paths, scores, uploaded_image, search_params, database):
    """Format search results into API response."""
    return {
        'paths': [r.replace(database.database_path, '', 1) for r in paths],
        'scores': scores,
        'filenames': [os.path.basename(r) for r in paths],
        'search_info': {
            'text_query': bool(search_params['query']),
            'image_query': bool(uploaded_image),
            'ocr_query': bool(search_params['ocr_text']),
            'object_filters': len(search_params['objects']),
            'models_used': search_params['models']
        }
    } 