import os
import numpy as np
from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel
from models.openclip import OpenCLIP
from backend.app.utils.generate_id import generate_id

class Qdrant:
    def __init__(self, host, port, caption_model=BGEM3FlagModel('BAAI/bge-m3', use_fp16=True), openclip_model=OpenCLIP('ViT-B-16', pretrained='dfn2b')):
        self.client = QdrantClient(host=host, port=port)
        self.caption_model = caption_model
        self.openclip_model = openclip_model
        
    def is_collection_exists(self, collection_name):
        """Check if collection exists in Qdrant"""
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
        
    def generate_caption_embeddings(self, text):
        return self.caption_model.encode(
            [text], 
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True
        )
        
    def create_caption_collection(self, collection_name):
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
    
    def insert_to_caption_collection(self, embeddings, collection_name):
        for embedding in embeddings:
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
                        id=generate_id(keyframe),
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
        
    def search_caption(self, search_query, collection_name, limit=100, prefetch_limit=300):
        # Generate caption embeddings for the query
        query_outputs = self.generate_caption_embeddings(search_query)
        
        dense_vec = query_outputs["dense_vecs"][0]
        sparse_vec = query_outputs["lexical_weights"][0]
        colbert_vec = query_outputs["colbert_vecs"][0]
        
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
        points = self.client.query_points(
            collection_name,
            prefetch=prefetch,
            query=colbert_vec,
            using="colbert",
            with_payload=True,
            limit=limit,
        ).points
        
        keyframes = [point.payload["keyframe"] for point in points]              
        
        return keyframes

    def generate_openclip_embeddings(self, text, image):
        if text:
            return self.openclip_model.encode_text(text)
        elif image:
            return self.openclip_model.encode_image(image)

    def create_openclip_collection(self, collection_name):
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=512,
                    distance=models.Distance.COSINE
                )
            }
        )

    def insert_to_openclip_collection(self, embeddings, collection_name):
        for embedding in embeddings:
            keyframe = embedding["keyframe"]
            embedded_vector = embedding["embedded_vector"]
            keyframe_name = os.path.splitext(keyframe)[0]
            parts = keyframe_name.split("_")
            batch_id = parts[0]
            video_id = parts[1]
            frame_id = parts[2]

            self.client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=generate_id(keyframe),
                        payload={
                            "keyframe": keyframe,
                            "batch_id": batch_id,
                            "video_id": video_id,
                            "frame_id": int(frame_id)
                        },
                        vector={
                            "dense": embedded_vector
                        }
                    )
                ]
            )

    def create_filter(self, include_batch_ids=None, exclude_batch_ids=None):
        should, must_not = [], []

        if include_batch_ids:
            should.append(models.FieldCondition(key="batch_id", match=models.MatchAny(any=include_batch_ids)))
        if exclude_batch_ids:
            must_not.append(models.FieldCondition(key="batch_id", match=models.MatchAny(any=exclude_batch_ids)))
        
        if not should and not must_not:
            return None
        return models.Filter(should=should, must_not=must_not)

    def search_openclip(self, text, image, collection_name, limit=100, include_batch_ids=None, exclude_batch_ids=None):
        dense_vec = self.generate_openclip_embeddings(text=text, image=image)

        points = self.client.query_points(
            collection_name,
            query=dense_vec,
            using="dense",
            with_payload=True,
            query_filter=self.create_filter(include_batch_ids=include_batch_ids, exclude_batch_ids=exclude_batch_ids),
            limit=limit,
        ).points
        
        keyframes = [point.payload["keyframe"] for point in points]
        return keyframes