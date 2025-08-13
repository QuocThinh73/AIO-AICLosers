import os
import json
from PIL import Image


def get_image_embedder(backbone, pretrained):
    from models.openclip import OpenCLIP
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_embedder = OpenCLIP(backbone, pretrained, device=device)
    return image_embedder

def process_video(keyframe_dir, output_embedded_vector_path, image_embedder):
    embedded_vectors = []
    for keyframe_name in sorted(os.listdir(keyframe_dir)):
        keyframe_path = os.path.join(keyframe_dir, keyframe_name)
        image = Image.open(keyframe_path).convert("RGB")
        embedded_vector = image_embedder.encode_image(image)

        embedded_vectors.append({
            "keyframe": keyframe_name,
            "embedded_vector": embedded_vector.tolist()
        })

    with open(output_embedded_vector_path, "w") as f:
        json.dump(embedded_vectors, f)
    

def embed_image(input_keyframe_dir, input_mapping_json, output_embedded_vector_dir, mode, backbone, pretrained, lesson_name=None, video_name=None):
    os.makedirs(output_embedded_vector_dir, exist_ok=True)

    image_embedder = get_image_embedder(backbone, pretrained)

    if mode == "all":
        pass
    elif mode == "lesson":
        pass
    elif mode == "video":
        pass