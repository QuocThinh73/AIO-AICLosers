import os
import json
from pathlib import Path

def build_mapping_json(image_paths, output_json="database/id2path.json"):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    items = []
    for i, p in enumerate(image_paths):
        posix_path = Path(p).as_posix()
        items.append({"id": i, "path": posix_path})
    
    data = {
        "total": len(items),
        "items": items
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import glob
    image_paths = glob.glob("data/keyframes/*.jpg")
    build_mapping_json(image_paths)
