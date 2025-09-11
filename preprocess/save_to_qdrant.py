import os
from typing import List, Dict
from backend.app.database.qdrant import Qdrant
from preprocess.utils import load_json


def _build_items(emb_path: str, cap_path: str) -> List[Dict]:
    embs = load_json(emb_path)
    print(f"Loaded {len(embs)} embedded vectors")
    caps = load_json(cap_path)
    print(f"Loaded {len(caps)} captions")
    m = {}
    for c in caps:
        k = c["keyframe"]
        cap = c.get("caption", "")
        b = os.path.basename(k)
        s = os.path.splitext(b)[0]
        m[k] = cap
        m[b] = cap
        m[s] = cap
    out = []
    for e in embs:
        k = e["keyframe"]
        vec = e.get("embedded_vector")
        b = os.path.basename(k)
        s = os.path.splitext(b)[0]
        out.append({"keyframe": k, "embedded_vector": vec,
                   "caption": (m.get(k) or m.get(b) or m.get(s) or "")})
    print(f"Processed {len(out)} items")
    return out


def process_video(emb_path, cap_path, qdrant_client, collection_name, batch_points):
    print(f"Processing video {emb_path} and {cap_path}")
    items = _build_items(emb_path, cap_path)
    qdrant_client.insert_to_collection(
        items=items, collection_name=collection_name, batch_points=batch_points)


def save_to_qdrant(input_embedded_vector_dir, input_caption_dir, mode, lesson_name=None, batch_points=64, collection_name="AIC2025"):
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
                              q, collection_name, batch_points)
    else:  # mode == "lesson"
        emb_dir = os.path.join(input_embedded_vector_dir, lesson_name)
        for video in os.listdir(emb_dir):
            if not video.endswith("_embedded_vector.json"):
                continue
            cap = video.replace("_embedded_vector.json", "_caption.json")
            process_video(os.path.join(emb_dir, video), os.path.join(input_caption_dir, lesson_name, cap),
                          q, collection_name, batch_points)
