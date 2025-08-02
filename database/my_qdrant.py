from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel
import json
import os
from app.config import MAPPING_JSON

class Qdrant:
    def __init__(self, host="localhost", port=6333, model=BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)):
        self.client = QdrantClient(host=host, port=port)
        self.model = model
        self.id2path = self.load_mapping(MAPPING_JSON)
        
    def load_mapping(self, mapping_json):
        """Load id2path mapping from JSON file"""
        with open(mapping_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        items = data.get("items", [])
        return {item["id"]: item["path"] for item in items}
        
    def get_keyframe_name(self, path):
        """Extract keyframe name from path"""
        return os.path.basename(path)
        
    def is_collection_exists(self, collection_name):
        return self.client.collection_exists(collection_name)
        
    def create_sparse_vector(self, sparse_data):
        """Convert BGE-M3 sparse output to Qdrant sparse vector format"""
        sparse_indices = []
        sparse_values = []
        
        for key, value in sparse_data.items():
            # Only process positive values
            if float(value) > 0:
                # Handle string keys
                if isinstance(key, str):
                    if key.isdigit():
                        key = int(key)
                    else:
                        continue
                    
                sparse_indices.append(key)
                sparse_values.append(float(value))
        
        return models.SparseVector(
            indices=sparse_indices,
            values=sparse_values
        )
        
    def generate_embeddings(self, text):
        return self.model.encode(
            [text], 
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True
        )
        
    def create_qdrant_collection(self, collection_name):
        self.client.create_collection(
            collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE
            ),
            "colbert": models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=True
                )
            )
        },
    )
    
    def insert_to_qdrant(self, embeddings, collection_name):
        for embedding in embeddings:
            point_id = embedding["point_id"]
            keyframe = embedding["keyframe"]
            caption = embedding["caption"]
            dense_vector = embedding["dense_vector"]
            colbert_vectors = embedding["colbert_vectors"]
            sparse_data = embedding["sparse_weights"]

            # Convert sparse weights to Qdrant format
            qdrant_sparse = self.create_sparse_vector(sparse_data)
            
            # Insert into Qdrant
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        payload={
                            "keyframe": keyframe,
                            "caption": caption
                        },
                        vector={
                            "dense": dense_vector,
                            "colbert": colbert_vectors,
                            "sparse": qdrant_sparse
                        }
                    )
                ]
            )
        
    def search(self, search_query, collection_name, limit=100, prefetch_limit=300):
        # Generate embeddings for the query
        query_outputs = self.model.encode(
            [search_query],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True
        )
        
        dense_vec = query_outputs["dense_vecs"][0]
        sparse_vec = query_outputs["lexical_weights"][0]
        colbert_vec = query_outputs["colbert_vecs"][0]
        
        # Convert sparse vector to Qdrant format
        qdrant_sparse = self.create_sparse_vector(sparse_vec)
        
        # Set up prefetch for hybrid search
        prefetch = [
            models.Prefetch(
                query=qdrant_sparse,
                using="sparse",
                limit=prefetch_limit),
            models.Prefetch(
                query=dense_vec,
                using="dense",
                limit=prefetch_limit)
        ]
        
        # Perform reranking with ColBERT
        results = self.client.query_points(
            collection_name,
            prefetch=prefetch,
            query=colbert_vec,
            using="colbert",
            with_payload=True,
            limit=limit,
        )["results"]["points"]
        
        indices, scores = zip(*[(point.id, point.score) for point in results])                   
        paths = [self.id2path[int(idx)] for idx in indices]
    
        return scores, indices, paths