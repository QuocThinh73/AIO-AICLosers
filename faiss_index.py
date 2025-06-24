import os
import faiss
import numpy as np
import json
from PIL import Image
from app.config import DATABASE_FOLDER

class Faiss:
    def __init__(self, model):   
        self.model = model
        self.embeddings = None
        self.id2path = None

    def load(self, index_path, mapping_json):
        # Load index
        self.embeddings = faiss.read_index(index_path)

         # Load mapping JSON
        with open(mapping_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        items = data.get("items", [])
        self.id2path = {item["id"]: item["path"] for item in items}

    def build(self, mapping_json, model_name, output_dir=DATABASE_FOLDER):
        os.makedirs(output_dir, exist_ok=True)

        # Load mapping JSON
        with open(mapping_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        total = data.get("total")
        items = data.get("items", [])
        id2path = {item["id"]: item["path"] for item in items}

        image_paths = [id2path[i] for i in range(total)]

        embeddings = []
        for path in image_paths:
          with Image.open(path).convert('RGB') as img:
            emb = self.model.encode_image(img)
          embeddings.append(emb)
            
        all_emb = np.vstack(embeddings).astype(np.float32)

        idx = faiss.IndexFlatIP(all_emb.shape[1])
        idx.add(all_emb)

        # Save
        embeddings_path = os.path.join(output_dir, f"{model_name}_faiss.bin")
        faiss.write_index(idx, embeddings_path)

    
    def text_search(self, query, top_k=5, return_scores=True):
        # Encode the query
        query_embedding = self.model.encode_text(query)
        
        # Ensure the embedding is in the right format
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search the index
        scores, indices = self.embeddings.search(query_embedding, top_k)
        
        # Get the image paths for the results
        paths = [self.id2path[int(idx)] for idx in indices[0]]
        
        if return_scores:
            return scores[0].tolist(), indices[0].tolist(), paths
        else:
            return paths
    
    def image_search(self, query_image, top_k=5, return_scores=True):   
        # Load the image if a path was provided
        if isinstance(query_image, str):
            query_image = Image.open(query_image).convert('RGB')
            
        # Encode the query image
        query_embedding = self.model.encode_image(query_image)
        
        # Ensure the embedding is in the right format
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search the index
        scores, indices = self.embeddings.search(query_embedding, top_k)
        
        # Get the image paths for the results
        paths = [self.id2path[int(idx)] for idx in indices[0]]
        
        if return_scores:
            return scores[0].tolist(), indices[0].tolist(), paths
        else:
            return paths