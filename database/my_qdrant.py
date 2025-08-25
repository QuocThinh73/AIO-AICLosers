from qdrant_client import QdrantClient, models
import json
import os
import numpy as np

class Qdrant:
    def __init__(self, host="localhost", port=6333, model=None):
        """
        Initialize Qdrant client for OpenCLIP embeddings
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            model: OpenCLIP model instance for embedding generation
        """
        self.client = QdrantClient(host=host, port=port)
        self.model = model
        
    def is_collection_exists(self, collection_name):
        """Check if collection exists in Qdrant"""
        return self.client.collection_exists(collection_name)
        
    def create_collection(self, collection_name, vector_size=512):
        """
        Create a new collection in Qdrant
        
        Args:
            collection_name: Name of the collection
            vector_size: Size of the embedding vectors (default: 512 for OpenCLIP)
        """
        if not self.is_collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            return True
        return False
    
    def batch_upload_points(self, embeddings_data, collection_name):
        """
        Upload a batch of points to Qdrant
        
        Args:
            embeddings_data: List of dictionaries with embedding data 
                             Each dict should have: id, vector, keyframe, path
            collection_name: Name of the collection to upload to
        """
        points = []
        
        for data in embeddings_data:
            points.append(
                models.PointStruct(
                    id=data["id"],
                    vector=data["vector"].tolist() if isinstance(data["vector"], np.ndarray) else data["vector"],
                    payload={
                        "keyframe": data["keyframe"],
                        "path": data["path"]
                    }
                )
            )
        
        if points:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
    
    def search_by_image(self, image, collection_name, limit=10):
        """
        Search for similar images in the collection
        
        Args:
            image: PIL Image object
            collection_name: Name of the collection to search in
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries with search results
        """
        if not self.model:
            raise ValueError("Model not provided. Cannot encode image.")
            
        # Encode image using OpenCLIP
        vector = self.model.encode_image(image)
        
        # Search in Qdrant
        results = self.client.search(
            collection_name=collection_name,
            query_vector=vector.tolist() if isinstance(vector, np.ndarray) else vector,
            limit=limit
        )
        
        # Format results
        formatted_results = []
        for point in results:
            formatted_results.append({
                "id": point.id,
                "score": float(point.score),
                "keyframe": point.payload.get("keyframe", ""),
                "path": point.payload.get("path", "")
            })
            
        return formatted_results
    
    def search_by_text(self, text, collection_name, limit=10):
        """
        Search for images matching text in the collection
        
        Args:
            text: Text query
            collection_name: Name of the collection to search in
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries with search results
        """
        if not self.model:
            raise ValueError("Model not provided. Cannot encode text.")
            
        # Encode text using OpenCLIP
        vector = self.model.encode_text(text)
        
        # Search in Qdrant
        results = self.client.search(
            collection_name=collection_name,
            query_vector=vector.tolist() if isinstance(vector, np.ndarray) else vector,
            limit=limit
        )
        
        # Format results
        formatted_results = []
        for point in results:
            formatted_results.append({
                "id": point.id,
                "score": float(point.score),
                "keyframe": point.payload.get("keyframe", ""),
                "path": point.payload.get("path", "")
            })
            
        return formatted_results
    
    def search(self, query, collection_name, limit=10):
        """
        Search for images matching query text in the collection
        This is an alias for search_by_text for compatibility
        
        Args:
            query: Text query
            collection_name: Name of the collection to search in
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries with search results
        """
        return self.search_by_text(query, collection_name, limit)
    
    def load_mapping(self, mapping_file):
        """
        Load mapping between IDs and paths from file
        
        Args:
            mapping_file: Path to the mapping JSON file
        
        Returns:
            Dictionary with mapping data
        """
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading mapping file: {e}")
            return {}
            
    def delete_collection(self, collection_name):
        """
        Delete a collection from Qdrant
        
        Args:
            collection_name: Name of the collection to delete
        
        Returns:
            Boolean indicating success
        """
        if self.is_collection_exists(collection_name):
            self.client.delete_collection(collection_name)
            return True
        return False
