import os
import sys
import json
from PIL import Image
import glob
import importlib
from tqdm import tqdm

def ensure_faiss_dependencies():
    dependencies = ["faiss-cpu", "open_clip_torch", "numpy"]
    
    for dep in dependencies:
        try:
            if dep == "faiss-cpu":
                import faiss
            elif dep == "open_clip_torch":
                import open_clip
            elif dep == "numpy":
                import numpy
        except ImportError:
            print(f"Đang cài đặt {dep}...")
            import subprocess
            if dep == "faiss-cpu":
                subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu==1.11.0"])
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"Đã cài đặt {dep}")

def load_model(backbone="ViT-B-16", pretrained="dfn2b"):
    try:
        ensure_faiss_dependencies()
        
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models.openclip import OpenCLIP
        
        print(f"Đang tải model OpenCLIP với backbone={backbone}, pretrained={pretrained}")
        model = OpenCLIP(backbone=backbone, pretrained=pretrained)
        print("Đã tải model thành công")
        return model
        
    except Exception as e:
        print(f"Lỗi khi tải model: {str(e)}")
        return None

def save_embeddings_faiss(keyframe_dir, output_dir, backbone="ViT-B-16", pretrained="dfn2b"):
    try:
        ensure_faiss_dependencies()
        
        os.makedirs(output_dir, exist_ok=True)
        
        model = load_model(backbone, pretrained)
        if model is None:
            return {"status": "error", "message": "Không thể tải model"}
        
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from database.my_faiss import Faiss
        my_faiss = Faiss(model=model)
        
        mapping_path = os.path.join(output_dir, "id2path.json")
        if not os.path.exists(mapping_path):
            print("Tạo file mapping id2path.json mới")
            create_mapping_file(keyframe_dir, mapping_path)
        
        my_faiss.load_mapping(mapping_json=mapping_path)
        
        model_name = f"OpenCLIP_{backbone}_{pretrained}"
        print(f"Đang tạo và lưu embeddings với model {model_name}")
        my_faiss.build(model_name=model_name, output_dir=output_dir)
        
        embeddings_path = os.path.join(output_dir, f"{model_name}_embeddings.bin")
        if os.path.exists(embeddings_path):
            print(f"Đã lưu embeddings thành công tại {embeddings_path}")
            return {"status": "success", "message": f"Đã tạo và lưu embeddings thành công tại {embeddings_path}"}
        else:
            return {"status": "error", "message": "Không thể lưu embeddings"}
        
    except Exception as e:
        print(f"Lỗi khi tạo embeddings: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Lỗi: {str(e)}"}

def create_mapping_file(keyframe_dir, output_path):
    try:
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_paths.extend(glob.glob(os.path.join(keyframe_dir, "**", ext), recursive=True))
        
        if not image_paths:
            print(f"Không tìm thấy file hình ảnh nào trong {keyframe_dir}")
            return
        
        items = []
        for i, path in enumerate(tqdm(image_paths, desc="Tạo mapping")):
            items.append({
                "id": i,
                "path": path
            })
        
        mapping = {"items": items}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        
        print(f"Đã tạo mapping cho {len(items)} hình ảnh và lưu vào {output_path}")
        
    except Exception as e:
        print(f"Lỗi khi tạo mapping: {str(e)}")
