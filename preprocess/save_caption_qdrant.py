import os
import sys
import json
import glob
from tqdm import tqdm

def ensure_qdrant_dependencies():

    dependencies = ["qdrant-client", "FlagEmbedding"]
    
    for dep in dependencies:
        try:
            if dep == "qdrant-client":
                from qdrant_client import QdrantClient, models
            elif dep == "FlagEmbedding":
                from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            print(f"Đang cài đặt {dep}...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"Đã cài đặt {dep}")

def load_caption_files(caption_folder):
    all_captions = []
    
    caption_files = glob.glob(os.path.join(caption_folder, "**", "*.json"), recursive=True)
    
    for caption_file in tqdm(caption_files, desc="Đang tải file caption"):
        print(f"Đang tải {caption_file}")
        try:
            with open(caption_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for item in data:
                keyframe = item['keyframe']
                caption = item['caption']
                
                all_captions.append({
                    "keyframe": keyframe,
                    "caption": caption
                })
        except Exception as e:
            print(f"Lỗi khi đọc file {caption_file}: {str(e)}")
    
    return all_captions

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

def save_captions_qdrant(caption_dir, keyframe_dir, output_dir, collection_name="captions"):
    try:
        ensure_qdrant_dependencies()
        os.makedirs(output_dir, exist_ok=True)
        
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from database.my_qdrant import Qdrant
        
        qdrant = Qdrant()
        
        if not qdrant.is_collection_exists(collection_name):
            print(f"Tạo collection '{collection_name}'...")
            qdrant.create_qdrant_collection(collection_name)
        else:
            print(f"Collection '{collection_name}' đã tồn tại.")
        
        mapping_path = os.path.join(output_dir, "id2path.json")
        if not os.path.exists(mapping_path):
            print("Tạo file mapping id2path.json mới")
            create_mapping_file(keyframe_dir, mapping_path)
            
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
            items = mapping_data.get("items", [])
            id2path = {item["id"]: item["path"] for item in items}
        except Exception as e:
            return {"status": "error", "message": f"Lỗi khi đọc file mapping: {str(e)}"}
            
        print("Tải file caption...")
        captions_data = load_caption_files(caption_dir)
        print(f"Đã tải {len(captions_data)} captions từ {caption_dir}")
        
        if len(captions_data) == 0:
            return {"status": "error", "message": "Không tìm thấy dữ liệu caption nào"}
        
        print("Xử lý embeddings và lưu vào Qdrant...")
        count = 0
        
        for caption_data in tqdm(captions_data, desc="Đang lưu vào Qdrant"):
            keyframe = caption_data['keyframe']
            caption = caption_data['caption']
            
            point_id = None
            for pid, path in id2path.items():
                if os.path.basename(path) == keyframe:
                    point_id = pid
                    break
            
            if point_id is None:
                print(f"Cảnh báo: Không tìm thấy keyframe {keyframe} trong mapping, bỏ qua...")
                continue
            
            try:
                embedding_output = qdrant.generate_embeddings(caption)
                
                dense_vector = embedding_output["dense_vecs"][0]
                colbert_vectors = embedding_output["colbert_vecs"][0]
                sparse_weights = embedding_output["lexical_weights"][0]

                embedding_data = {
                    "point_id": point_id,
                    "keyframe": keyframe,
                    "caption": caption,
                    "dense_vector": dense_vector,
                    "colbert_vectors": colbert_vectors,
                    "sparse_weights": sparse_weights
                }
                
                qdrant.insert_to_qdrant([embedding_data], collection_name)
                count += 1
            except Exception as e:
                print(f"Lỗi khi xử lý caption cho keyframe {keyframe}: {str(e)}")
                continue
        
        if count > 0:
            return {"status": "success", "message": f"Đã lưu thành công {count} caption vào Qdrant"}
        else:
            return {"status": "error", "message": "Không có dữ liệu nào được lưu vào Qdrant"}
        
    except Exception as e:
        print(f"Lỗi khi lưu caption vào Qdrant: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Lỗi: {str(e)}"}