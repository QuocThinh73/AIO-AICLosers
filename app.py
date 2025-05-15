import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict, Any, Union
import numpy as np
from PIL import Image
import io
import json

from vlm import CLIPModel
from translation import Translation
from faiss_index import FaissIndex

# Initialize the app
app = FastAPI(
    title="Text-to-Video Search API",
    description="API for searching videos by text description using Vision-Language Models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
MODEL = None
TRANSLATOR = None
FAISS_INDEX = None

@app.on_event("startup")
async def startup_event():
    """Initialize models and index on startup."""
    global MODEL, TRANSLATOR, FAISS_INDEX
    
    # Initialize CLIP model
    print("Loading CLIP model...")
    MODEL = CLIPModel()
    
    # Initialize translator
    print("Initializing translator...")
    TRANSLATOR = Translation(backend="google")
    
    # Initialize FAISS index if available
    faiss_index_dir = "faiss_index"
    index_path = os.path.join(faiss_index_dir, "faiss_index.bin")
    id2path_path = os.path.join(faiss_index_dir, "id2path.pkl")
    
    if os.path.exists(index_path) and os.path.exists(id2path_path):
        print(f"Loading FAISS index from {index_path}...")
        FAISS_INDEX = FaissIndex(
            index_path=index_path,
            id2path_path=id2path_path,
            model=MODEL,
            translator=TRANSLATOR
        )
        
        # Print index stats
        stats = FAISS_INDEX.get_stats()
        print(f"Loaded index with {stats['num_vectors']} vectors of dimension {stats['dimension']}")
    else:
        print("FAISS index not found. Please build the index first.")


# Mount static files for serving keyframes
if os.path.exists("data/keyframes"):
    app.mount("/keyframes", StaticFiles(directory="data/keyframes"), name="keyframes")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Text-to-Video Search API",
        "docs_url": "/docs",
        "status": "active"
    }


@app.get("/status")
async def status():
    """Check status of models and index."""
    global MODEL, TRANSLATOR, FAISS_INDEX
    
    index_status = "not_loaded"
    index_stats = {}
    
    if FAISS_INDEX is not None:
        index_status = "loaded"
        index_stats = FAISS_INDEX.get_stats()
    
    return {
        "status": "active",
        "models": {
            "vlm": {
                "name": "CLIP",
                "status": "loaded" if MODEL is not None else "not_loaded"
            },
            "translator": {
                "name": "Translation",
                "backend": TRANSLATOR.backend if TRANSLATOR is not None else None,
                "status": "loaded" if TRANSLATOR is not None else "not_loaded"
            }
        },
        "index": {
            "status": index_status,
            "stats": index_stats
        }
    }


@app.get("/search/text")
async def search_by_text(
    query: str, 
    top_k: int = 10,
    return_videos: bool = True
):
    """Search for videos by text description.
    
    Args:
        query (str): Text query (in English or Vietnamese)
        top_k (int): Number of results to return
        return_videos (bool): Whether to return video paths
        
    Returns:
        Dict: Search results including scores, indices, and paths
    """
    global FAISS_INDEX
    
    if FAISS_INDEX is None:
        raise HTTPException(status_code=500, detail="FAISS index not loaded")
    
    # Perform search
    scores, indices, paths = FAISS_INDEX.text_search(query, top_k=top_k)
    
    # Get video paths if requested
    video_paths = None
    if return_videos:
        video_paths = []
        for path in paths:
            video_path = FAISS_INDEX.get_video_for_frame(path)
            video_paths.append(video_path)
    
    # Format results
    results = []
    for i in range(len(paths)):
        result = {
            "score": float(scores[i]),
            "index": int(indices[i]),
            "keyframe_path": paths[i],
            "keyframe_url": f"/keyframes/{os.path.basename(os.path.dirname(paths[i]))}/{os.path.basename(paths[i])}"
        }
        
        if return_videos and video_paths is not None:
            result["video_path"] = video_paths[i]
            
        results.append(result)
    
    return {
        "query": query,
        "translated_query": TRANSLATOR(query) if TRANSLATOR is not None else query,
        "results": results
    }


@app.post("/search/image")
async def search_by_image(
    file: UploadFile = File(...),
    top_k: int = 10,
    return_videos: bool = True
):
    """Search for videos by image.
    
    Args:
        file (UploadFile): Image file
        top_k (int): Number of results to return
        return_videos (bool): Whether to return video paths
        
    Returns:
        Dict: Search results including scores, indices, and paths
    """
    global FAISS_INDEX
    
    if FAISS_INDEX is None:
        raise HTTPException(status_code=500, detail="FAISS index not loaded")
    
    # Read and validate the uploaded image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    # Perform search
    scores, indices, paths = FAISS_INDEX.image_search(image, top_k=top_k)
    
    # Get video paths if requested
    video_paths = None
    if return_videos:
        video_paths = []
        for path in paths:
            video_path = FAISS_INDEX.get_video_for_frame(path)
            video_paths.append(video_path)
    
    # Format results
    results = []
    for i in range(len(paths)):
        result = {
            "score": float(scores[i]),
            "index": int(indices[i]),
            "keyframe_path": paths[i],
            "keyframe_url": f"/keyframes/{os.path.basename(os.path.dirname(paths[i]))}/{os.path.basename(paths[i])}"
        }
        
        if return_videos and video_paths is not None:
            result["video_path"] = video_paths[i]
            
        results.append(result)
    
    return {
        "query_type": "image",
        "query_filename": file.filename,
        "results": results
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 