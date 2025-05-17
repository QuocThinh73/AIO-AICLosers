import os
import faiss
import numpy as np
import pickle
from typing import List, Dict, Tuple, Any
from PIL import Image
import time
from tqdm import tqdm

from vlm import BaseVLM

def build_faiss_index(
    image_paths: List[str], 
    model: BaseVLM, 
    output_dir: str = "faiss_index",
    batch_size: int = 32,
    verbose: bool = True
) -> Tuple[faiss.Index, Dict[int, str]]:
    """Build a FAISS index from a list of image paths using a VLM model.
    
    Args:
        image_paths (List[str]): List of paths to images to index
        model (BaseVLM): Vision-Language Model to use for encoding images
        output_dir (str): Directory to save the index and id-to-path mapping
        batch_size (int): Batch size for processing images
        verbose (bool): Whether to show progress bars and logs
        
    Returns:
        Tuple[faiss.Index, Dict[int, str]]: The FAISS index and id-to-path mapping
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Create a dictionary mapping IDs to image paths
    id2path = {i: path for i, path in enumerate(image_paths)}
    
    # Calculate embeddings for all images
    if verbose:
        print(f"Computing embeddings for {len(image_paths)} images...")
    
    embeddings = []
    
    # Process images in batches
    for i in tqdm(range(0, len(image_paths), batch_size), disable=not verbose):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        # Load images
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                batch_images.append(img)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                # Use a placeholder instead
                batch_images.append(Image.new('RGB', (224, 224), color='black'))
        
        # Encode images
        batch_embeddings = model.encode_batch_images(batch_images)
        embeddings.append(batch_embeddings)
        
        # Close images to save memory
        for img in batch_images:
            img.close()
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(embeddings).astype(np.float32)
    
    if verbose:
        print(f"Building FAISS index with dimension {all_embeddings.shape[1]}...")
    
    # Create and train the index
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product similarity (cosine)
    index.add(all_embeddings)
    
    # Save the index and ID mapping
    index_path = os.path.join(output_dir, "faiss_index.bin")
    mapping_path = os.path.join(output_dir, "id2path.pkl")
    
    faiss.write_index(index, index_path)
    with open(mapping_path, 'wb') as f:
        pickle.dump(id2path, f)
    
    if verbose:
        print(f"FAISS index saved to {index_path}")
        print(f"ID to path mapping saved to {mapping_path}")
    
    return index, id2path