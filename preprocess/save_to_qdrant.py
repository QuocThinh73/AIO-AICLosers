import os
from typing import List, Dict
from backend.app.database.qdrant import Qdrant
from preprocess.utils import load_json


def _build_items(emb_path: str, cap_path: str) -> List[Dict]:
    embeddings_json = load_json(emb_path)
    captions_json = load_json(cap_path)
    captions_map = {}
    for caption in captions_json:
        keyframe = caption["keyframe"]
        caption = caption["caption"]
        captions_map[keyframe] = caption
    results = []
    for embedding in embeddings_json:
        keyframe = embedding["keyframe"]
        embedded_vector = embedding["embedded_vector"]
        results.append({
            "keyframe": keyframe,
            "embedded_vector": embedded_vector,
            "caption": captions_map[keyframe]
        })
    return results


def process_video(emb_path, cap_path, qdrant_client, collection_name, batch_size):
    print(f"Processing video {os.path.basename(emb_path).split('.')[0]}")
    items = _build_items(emb_path, cap_path)
    qdrant_client.insert_to_collection(
        items=items, collection_name=collection_name, batch_size=batch_size)


def save_to_qdrant(input_embedded_vector_dir, input_caption_dir, mode, batch_size=64, lesson_name=None, collection_name="AIC2025"):
    q = Qdrant(host="localhost", port=6333)
    if not q.is_collection_exists(collection_name):
        q.create_collection(collection_name)
    if mode == "all":
        for lesson in os.listdir(input_embedded_vector_dir):
            emb_dir = os.path.join(input_embedded_vector_dir, lesson)
            for video in os.listdir(emb_dir):
                if not video.endswith("_embedded_vector.json"):
                    continue
                cap = video.replace("_embedded_vector.json", "_caption.json")
                process_video(os.path.join(emb_dir, video), os.path.join(input_caption_dir, lesson, cap),
                              q, collection_name, batch_size)
    elif mode == "lesson":
        emb_dir = os.path.join(input_embedded_vector_dir, lesson_name)
        for video in os.listdir(emb_dir):
            if not video.endswith("_embedded_vector.json"):
                continue
            cap = video.replace("_embedded_vector.json", "_caption.json")
            process_video(os.path.join(emb_dir, video), os.path.join(input_caption_dir, lesson_name, cap),
                          q, collection_name, batch_size)
