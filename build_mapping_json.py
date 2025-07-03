import os
import json
from pathlib import Path
from utils import get_all_keyframe_paths

DATABASE_FOLDER = "database"
KEYFRAME_FOLDER = os.path.join(DATABASE_FOLDER, "keyframes")
MAPPING_JSON = "id2path.json"

def build_mapping_json(keyframe_paths, output_json=os.path.join(DATABASE_FOLDER, MAPPING_JSON)):
    """
    Build a mapping JSON file from list of image paths.
    
    Args:
        keyframe_paths: List of image file paths
        output_json: Output JSON file path
    """
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    items = []
    for i, p in enumerate(keyframe_paths):
        posix_path = Path(p).as_posix()
        items.append({"id": i, "path": posix_path})
    
    data = {
        "total": len(items),
        "items": items
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    keyframe_paths = get_all_keyframe_paths(KEYFRAME_FOLDER)
    build_mapping_json(keyframe_paths)
