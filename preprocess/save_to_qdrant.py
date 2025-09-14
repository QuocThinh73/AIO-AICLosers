import os
from typing import List, Dict
from backend.app.database.qdrant import Qdrant
from preprocess.utils import load_json


def _build_items(emb_path: str, cap_path: str, ocr_path: str) -> List[Dict]:
    embeddings_json = load_json(emb_path)
    captions_json = load_json(cap_path)
    ocr_json = load_json(ocr_path)
    captions_map = {}
    for caption in captions_json:
        keyframe = caption["keyframe"]
        caption = caption["caption"]
        captions_map[keyframe] = caption
    ocr_map = {}
    for ocr in ocr_json:
        keyframe = ocr["image"]
        ocr = [result for result in ocr["results"] if result["text"].lower(
        ) not in ["g1a", "uiay", "giay", "gia_"] and result["confidence"] > 0.4]
        ocr_map[keyframe] = ocr
    results = []
    for embedding in embeddings_json:
        keyframe = embedding["keyframe"]
        embedded_vector = embedding["embedded_vector"]
        results.append({
            "keyframe": keyframe,
            "embedded_vector": embedded_vector,
            "caption": captions_map[keyframe],
            "ocr": ocr_map[keyframe]
        })
    return results


def process_video(emb_path, cap_path, ocr_path, qdrant_client, collection_name, batch_size):
    print(f"Processing video {os.path.basename(emb_path).split('.')[0]}")
    items = _build_items(emb_path, cap_path, ocr_path)
    qdrant_client.insert_to_collection(
        items=items, collection_name=collection_name, batch_size=batch_size)


def save_to_qdrant(input_embedded_vector_dir, input_caption_dir, input_ocr_dir, mode, batch_size=1024, lesson_name=None, collection_name="AIC2025"):
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
                ocr = video.replace("_embedded_vector.json", "_ocr.json")
                process_video(os.path.join(emb_dir, video), os.path.join(input_caption_dir, lesson, cap), os.path.join(input_ocr_dir, lesson, ocr),
                              q, collection_name, batch_size)
    elif mode == "lesson":
        emb_dir = os.path.join(input_embedded_vector_dir, lesson_name)
        for video in os.listdir(emb_dir):
            if not video.endswith("_embedded_vector.json"):
                continue
            cap = video.replace("_embedded_vector.json", "_caption.json")
            ocr = video.replace("_embedded_vector.json", "_ocr.json")
            process_video(os.path.join(emb_dir, video), os.path.join(input_caption_dir, lesson_name, cap), os.path.join(input_ocr_dir, lesson_name, ocr),
                          q, collection_name, batch_size)
