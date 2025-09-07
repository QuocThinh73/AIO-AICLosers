import os
from backend.app.database.qdrant import Qdrant
from preprocess.utils import load_json

def process_video(embedded_vector_file_path, qdrant_client):
    embedded_vectors = load_json(embedded_vector_file_path)
    qdrant_client.insert_to_openclip_collection(embeddings=embedded_vectors, collection_name="openclip")

def save_embeddings_qdrant(input_embedded_vector_dir):
    qdrant_client = Qdrant(host="localhost", port=6333)
    if not qdrant_client.is_collection_exists(collection_name="openclip"):
        qdrant_client.create_openclip_collection(collection_name="openclip")

    for batch in os.listdir(input_embedded_vector_dir):
        for video in os.listdir(os.path.join(input_embedded_vector_dir, batch)):
            process_video(os.path.join(input_embedded_vector_dir, batch, video), qdrant_client)