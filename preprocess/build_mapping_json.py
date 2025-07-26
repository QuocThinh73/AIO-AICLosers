import os
import json
from pathlib import Path
import glob

def get_keyframe_paths(input_keyframe_dir):
    keyframe_paths = []
    for root, _, files in os.walk(input_keyframe_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                keyframe_paths.append(os.path.join(root, file))
    return sorted(keyframe_paths)

def build_mapping_json(input_keyframe_dir, output_mapping_json):
    os.makedirs(os.path.dirname(output_mapping_json), exist_ok=True)
    
    keyframe_paths = get_keyframe_paths(input_keyframe_dir)
    
    items = []
    for i, p in enumerate(keyframe_paths):
        posix_path = Path(p).as_posix()
        items.append({"id": i, "path": posix_path})
    
    data = {
        "total": len(items),
        "items": items
    }
    #Save
    with open(output_mapping_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Mapping JSON saved to {output_mapping_json} with {len(items)} items")
    return data