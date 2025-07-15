import os
import json
import glob
from tqdm import tqdm
from qdrant import Qdrant

DATABASE_FOLDER = "database"
CAPTION_FOLDER = os.path.join(DATABASE_FOLDER, "captions")
ID2PATH_FILE = os.path.join(DATABASE_FOLDER, "id2path.json")
COLLECTION_NAME = "captions"

def load_caption_files(caption_folder):
    """
    Load all caption files from the caption folder.
    
    Returns:
        list: List of all caption data with keyframe names and captions
    """
    all_captions = []
    
    # Find all JSON files in all subdirectories
    caption_files = glob.glob(os.path.join(caption_folder, "**", "*.json"), recursive=True)
    
    for caption_file in caption_files:
        print(f"Loading {caption_file}")
        with open(caption_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Add each caption with its keyframe name
        for item in data:
            keyframe = item['keyframe']
            caption = item['caption']
            
            all_captions.append({
                "keyframe": keyframe,
                "caption": caption
            })
    
    return all_captions

def process_embeddings(qdrant_client, captions_data, collection_name):
    """
    Process embeddings one by one and insert immediately.
    
    Args:
        qdrant_client: Qdrant client instance
        captions_data: List of caption data
        collection_name: Name of the Qdrant collection
    """
    for caption_data in tqdm(captions_data):
        keyframe = caption_data['keyframe']
        caption = caption_data['caption']
        
        # Find the ID for this keyframe from id2path mapping
        point_id = None
        for pid, path in qdrant_client.id2path.items():
            if qdrant_client.get_keyframe_name(path) == keyframe:
                point_id = pid
                break
        
        if point_id is None:
            print(f"Warning: Keyframe {keyframe} not found in ID mapping, skipping...")
            continue
        
        # Generate embeddings using BGE-M3 model
        embedding_output = qdrant_client.generate_embeddings(caption)
        
        # Extract vectors from the embedding output
        dense_vector = embedding_output["dense_vecs"][0]
        colbert_vectors = embedding_output["colbert_vecs"][0]
        sparse_weights = embedding_output["lexical_weights"][0]
        
        # Prepare embedding data for insertion
        embedding_data = {
            "point_id": point_id,
            "keyframe": keyframe,
            "caption": caption,
            "dense_vector": dense_vector,
            "colbert_vectors": colbert_vectors,
            "sparse_weights": sparse_weights
        }
        
        # Insert into Qdrant
        qdrant_client.insert_to_qdrant([embedding_data], collection_name)

def main():
    print("Initializing Qdrant client...")
    qdrant = Qdrant()
    
    # Create collection if it doesn't exist
    if not qdrant.is_collection_exists(COLLECTION_NAME):
        print(f"Creating collection '{COLLECTION_NAME}'...")
        qdrant.create_qdrant_collection(COLLECTION_NAME)
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    
    # Load all caption files
    print("Loading caption files...")
    captions_data = load_caption_files(CAPTION_FOLDER)
    print(f"Loaded {len(captions_data)} captions from {CAPTION_FOLDER}")
    
    # Process embeddings
    process_embeddings(qdrant, captions_data, COLLECTION_NAME)
    
    print("Successfully completed processing!")

if __name__ == "__main__":
    main()
    





