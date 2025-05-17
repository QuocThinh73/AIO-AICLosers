import os
import faiss
import numpy as np
import pickle
from PIL import Image

class FaissIndex:
    def __init__(self, model):   
        self.model = model

    @classmethod
    def load(self, index_path, id2path_path):
        # Load the FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load the ID to path mapping
        with open(id2path_path, 'rb') as f:
            self.id2path = pickle.load(f)

    @classmethod
    def build(self, image_paths, model, output_dir="faiss_index", batch_size=32):
        os.makedirs(output_dir, exist_ok=True)
        id2path = {i: path for i, path in enumerate(image_paths)}

        embeddings = []
        for path in image_paths:
            try:
                img = Image.open(path).convert('RGB')
            except Exception:
                img = Image.new('RGB', (224, 224), 'black')
            emb = model.encode_image(img)   # dùng phương thức của bạn
            embeddings.append(emb)
            img.close()
            
        all_emb = np.vstack(embeddings).astype(np.float32)

        idx = faiss.IndexFlatIP(all_emb.shape[1])
        idx.add(all_emb)

        # Save
        index_path = os.path.join(output_dir, "faiss_index.bin")
        map_path = os.path.join(output_dir, "id2path.pkl")
        faiss.write_index(idx, index_path)
        with open(map_path, 'wb') as f:
            pickle.dump(id2path, f)
        
    def text_search(self, query, top_k=5, return_scores=True):
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
    
    def image_search(self, query_image, top_k=5, return_scores=True):   
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
        
    def get_stats(self):
        return {
            "num_vectors": self.index.ntotal,
            "dimension": self.index.d,
            "num_images": len(self.id2path)
        }