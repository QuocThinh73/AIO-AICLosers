import os
import faiss
import numpy as np
import pickle
from typing import List, Dict, Tuple, Union, Optional
from PIL import Image
import time

from vlm import BaseVLM
from translation import Translation

class FaissIndex:
    """FAISS index for efficient image retrieval using text or image queries."""
    
    def __init__(
        self, 
        index_path: str,
        id2path_path: str,
        model: BaseVLM,
        translator: Optional[Translation] = None,
        top_k: int = 10
    ):
        """Initialize the FAISS index.
        
        Args:
            index_path (str): Path to the FAISS index file
            id2path_path (str): Path to the ID to image path mapping file
            model (BaseVLM): Vision-Language Model for encoding queries
            translator (Translation, optional): Translator for non-English queries
            top_k (int): Default number of results to return
        """
        # Load the FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load the ID to path mapping
        with open(id2path_path, 'rb') as f:
            self.id2path = pickle.load(f)
            
        self.model = model
        self.translator = translator
        self.top_k = top_k
        
    def text_search(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        return_scores: bool = True
    ) -> Union[List[str], Tuple[List[float], List[int], List[str]]]:
        """Search for images matching a text query.
        
        Args:
            query (str): Text query
            top_k (int, optional): Number of results to return (default: self.top_k)
            return_scores (bool): Whether to return similarity scores and indices
        
        Returns:
            If return_scores is True:
                Tuple[List[float], List[int], List[str]]: Similarity scores, indices, and image paths
            Else:
                List[str]: List of image paths
        """
        if top_k is None:
            top_k = self.top_k
            
        # Translate query if necessary
        if self.translator is not None:
            query = self.translator(query)
            
        # Encode the query
        query_embedding = self.model.encode_text(query)
        
        # Ensure the embedding is in the right format
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search the index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Get the image paths for the results
        paths = [self.id2path[int(idx)] for idx in indices[0]]
        
        if return_scores:
            return scores[0].tolist(), indices[0].tolist(), paths
        else:
            return paths
    
    def image_search(
        self, 
        query_image: Union[str, Image.Image], 
        top_k: Optional[int] = None,
        return_scores: bool = True
    ) -> Union[List[str], Tuple[List[float], List[int], List[str]]]:
        """Search for images similar to a query image.
        
        Args:
            query_image (Union[str, Image.Image]): Query image path or PIL Image
            top_k (int, optional): Number of results to return (default: self.top_k)
            return_scores (bool): Whether to return similarity scores and indices
        
        Returns:
            If return_scores is True:
                Tuple[List[float], List[int], List[str]]: Similarity scores, indices, and image paths
            Else:
                List[str]: List of image paths
        """
        if top_k is None:
            top_k = self.top_k
            
        # Load the image if a path was provided
        if isinstance(query_image, str):
            query_image = Image.open(query_image).convert('RGB')
            
        # Encode the query image
        query_embedding = self.model.encode_image(query_image)
        
        # Ensure the embedding is in the right format
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search the index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Get the image paths for the results
        paths = [self.id2path[int(idx)] for idx in indices[0]]
        
        if return_scores:
            return scores[0].tolist(), indices[0].tolist(), paths
        else:
            return paths
    
    def save(self, index_path: str, id2path_path: str):
        """Save the FAISS index and ID to path mapping.
        
        Args:
            index_path (str): Path to save the FAISS index
            id2path_path (str): Path to save the ID to path mapping
        """
        # Save the FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save the ID to path mapping
        with open(id2path_path, 'wb') as f:
            pickle.dump(self.id2path, f)
    
    @classmethod
    def load(
        cls, 
        index_path: str,
        id2path_path: str,
        model: BaseVLM,
        translator: Optional[Translation] = None,
        top_k: int = 10
    ) -> 'FaissIndex':
        """Load a FaissIndex from saved files.
        
        Args:
            index_path (str): Path to the FAISS index file
            id2path_path (str): Path to the ID to image path mapping file
            model (BaseVLM): Vision-Language Model for encoding queries
            translator (Translation, optional): Translator for non-English queries
            top_k (int): Default number of results to return
            
        Returns:
            FaissIndex: Loaded FAISS index
        """
        return cls(
            index_path=index_path,
            id2path_path=id2path_path,
            model=model,
            translator=translator,
            top_k=top_k
        )
        
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the index.
        
        Returns:
            Dict[str, int]: Dictionary of statistics
        """
        return {
            "num_vectors": self.index.ntotal,
            "dimension": self.index.d,
            "num_images": len(self.id2path)
        }