import os
from backend.app.database.qdrant import Qdrant
from preprocess.utils import load_json


def process_video(input_embedded_vector_path, input_caption_path, qdrant_client):
    embedded_vectors = load_json(input_embedded_vector_path)
    captions = load_json(input_caption_path)
    qdrant_client.insert_to_collection(
        embeddings=embedded_vectors, captions=captions, collection_name="AIC2025")


def save_to_qdrant(input_embedded_vector_dir, input_caption_dir, mode, lesson_name=None):
    if mode == "all":
        qdrant_client = Qdrant(host="localhost", port=6333)
        if not qdrant_client.is_collection_exists(collection_name="AIC2025"):
            qdrant_client.create_openclip_collection(collection_name="AIC2025")
        for lesson in os.listdir(input_embedded_vector_dir):
            for video in os.listdir(os.path.join(input_embedded_vector_dir, lesson)):
                process_video(os.path.join(input_embedded_vector_dir, lesson, video), os.path.join(
                    input_caption_dir, lesson, video), qdrant_client)
    elif mode == "lesson":
        qdrant_client = Qdrant(host="localhost", port=6333)
        if not qdrant_client.is_collection_exists(collection_name="AIC2025"):
            qdrant_client.create_openclip_collection(collection_name="AIC2025")
        for video in os.listdir(os.path.join(input_embedded_vector_dir, lesson_name)):
            process_video(os.path.join(input_embedded_vector_dir, lesson_name, video), os.path.join(
                input_caption_dir, lesson_name, video), qdrant_client)
