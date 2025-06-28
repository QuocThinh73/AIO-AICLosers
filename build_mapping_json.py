import os
import json
import glob
from pathlib import Path

DATABASE_FOLDER = "database"
MAPPING_JSON = "id2path.json"

def build_mapping_json(image_paths, output_json=os.path.join(DATABASE_FOLDER, MAPPING_JSON)):
    """
    Build a mapping JSON file from list of image paths.
    
    Args:
        image_paths: List of image file paths
        output_json: Output JSON file path
    """
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

def get_all_keyframe_paths():
    """
    Scan for all keyframe images in the L/V directory structure.
    Falls back to flat directory structure if no L directories found.
    
    Returns:
        list: Sorted list of image file paths
    """
    keyframes_base = os.path.join(DATABASE_FOLDER, "keyframes")
    image_paths = []
    
    # Method 1: Scan L/V subdirectory structure
    l_dirs = []
    if os.path.exists(keyframes_base):
        for item in os.listdir(keyframes_base):
            item_path = os.path.join(keyframes_base, item)
            if os.path.isdir(item_path) and item.startswith('L'):
                l_dirs.append(item)
    
    if l_dirs:
        # Scan L/V structure: database/keyframes/L01/V001/*.jpg
        for l_dir in sorted(l_dirs):
            l_path = os.path.join(keyframes_base, l_dir)
            if not os.path.isdir(l_path):
                continue
                
            for v_item in os.listdir(l_path):
                v_path = os.path.join(l_path, v_item)
                if os.path.isdir(v_path) and v_item.startswith('V'):
                    pattern = os.path.join(v_path, "*.jpg")
                    v_images = glob.glob(pattern)
                    image_paths.extend(v_images)
    else:
        # Method 2: Fallback to flat structure: database/keyframes/*.jpg
        pattern = os.path.join(keyframes_base, "*.jpg")
        image_paths = glob.glob(pattern)
    
    return sorted(image_paths)

if __name__ == "__main__":
    image_paths = get_all_keyframe_paths()
    build_mapping_json(image_paths)
