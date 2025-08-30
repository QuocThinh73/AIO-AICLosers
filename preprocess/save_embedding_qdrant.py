import os
import sys
import json
import uuid
from PIL import Image
import glob
from tqdm import tqdm

def ensure_dependencies():
    dependencies = ["qdrant-client", "open_clip_torch", "numpy"]
    
    for dep in dependencies:
        try:
            if dep == "qdrant-client":
                from qdrant_client import QdrantClient, models
            elif dep == "open_clip_torch":
                import open_clip
            elif dep == "numpy":
                import numpy
        except ImportError:
            print(f"Đang cài đặt {dep}...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"Đã cài đặt {dep}")

def load_openclip_model(backbone="ViT-B-16", pretrained="dfn2b"):
    try:
        ensure_dependencies()
        
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models.openclip import OpenCLIP
        
        print(f"Đang tải model OpenCLIP với backbone={backbone}, pretrained={pretrained}")
        model = OpenCLIP(backbone=backbone, pretrained=pretrained)
        print("Đã tải model OpenCLIP thành công")
        return model
    except Exception as e:
        print(f"Lỗi khi tải model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def save_embeddings_from_file(embedded_vector_file, keyframe_dir, output_dir, collection_name="image_embeddings_precomputed"):
    """
    Lưu embeddings từ file đã tạo trước đó vào Qdrant
    
    Args:
        embedded_vector_file: Đường dẫn đến file JSON chứa các vectors đã được tạo bởi embed_image
        keyframe_dir: Thư mục chứa các keyframes
        output_dir: Thư mục đầu ra
        collection_name: Tên collection trong Qdrant
        
    Returns:
        Dictionary chứa thông tin kết quả
    """
    try:
        ensure_dependencies()
        
        # Đảm bảo thư mục đầu ra tồn tại
        os.makedirs(output_dir, exist_ok=True)
        
        # Kiểm tra file embeddings tồn tại
        if not os.path.exists(embedded_vector_file):
            return {"status": "error", "message": f"File embeddings không tồn tại: {embedded_vector_file}"}
        
        # Load embeddings từ file
        print(f"Đang đọc file embeddings: {embedded_vector_file}")
        with open(embedded_vector_file, 'r') as f:
            embedded_vectors = json.load(f)
        
        print(f"Đã đọc {len(embedded_vectors)} embeddings từ file")
        
        # Import và khởi tạo Qdrant
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from database.my_qdrant_openclip import Qdrant
            # Khởi tạo client mà không cần model (chỉ để lưu)
            qdrant = Qdrant(model=None)
        except ImportError:
            return {"status": "error", "message": "Không thể import Qdrant"}
            
        # Tạo collection nếu chưa có
        if not qdrant.is_collection_exists(collection_name):
            print(f"Tạo collection '{collection_name}'...")
            qdrant.create_collection(collection_name)
        else:
            print(f"Collection '{collection_name}' đã tồn tại.")
        
        # Xử lý và lưu embeddings vào Qdrant
        count = 0
        batch_size = 10  # Xử lý theo batch để tối ưu
        current_batch = []
        keyframe_mapping = {}  # Lưu mapping giữa path và id
        
        for item in tqdm(embedded_vectors, desc="Đang lưu embeddings"):
            try:
                # Lấy thông tin từ item
                keyframe_name = item["keyframe"]
                embedded_vector = item["embedded_vector"]
                
                # Tìm đường dẫn đầy đủ của keyframe
                keyframe_path = ""
                for root, dirs, files in os.walk(keyframe_dir):
                    if keyframe_name in files:
                        keyframe_path = os.path.join(root, keyframe_name)
                        break
                        
                if not keyframe_path:
                    print(f"Không tìm thấy đường dẫn cho keyframe: {keyframe_name}")
                    continue
                
                # Tạo random UUID cho mỗi hình ảnh
                point_id = str(uuid.uuid4())
                
                # Lưu mapping
                keyframe_mapping[point_id] = {
                    "path": keyframe_path,
                    "keyframe": keyframe_name
                }
                
                # Chuẩn bị dữ liệu để lưu
                embedding_data = {
                    "id": point_id,
                    "keyframe": keyframe_name,
                    "path": keyframe_path,
                    "vector": embedded_vector
                }
                
                current_batch.append(embedding_data)
                
                # Nếu đủ batch size thì lưu vào Qdrant
                if len(current_batch) >= batch_size:
                    qdrant.batch_upload_points(current_batch, collection_name)
                    count += len(current_batch)
                    current_batch = []
                    print(f"Đã lưu {count}/{len(embedded_vectors)} embeddings")
            
            except Exception as e:
                print(f"Lỗi khi xử lý embedding cho {keyframe_name}: {str(e)}")
                continue
        
        # Lưu batch cuối cùng nếu còn
        if current_batch:
            qdrant.batch_upload_points(current_batch, collection_name)
            count += len(current_batch)
            print(f"Đã lưu {count}/{len(embedded_vectors)} embeddings")
        
        # Lưu mapping giữa ID và path
        mapping_path = os.path.join(output_dir, "keyframe_mapping.json")
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(keyframe_mapping, f, ensure_ascii=False, indent=2)
            
        print(f"Đã lưu mapping giữa ID và path vào {mapping_path}")
        
        if count > 0:
            return {
                "status": "success", 
                "message": f"Đã lưu {count} embeddings thành công vào Qdrant",
                "mapping_file": mapping_path
            }
        else:
            return {"status": "error", "message": "Không có embedding nào được lưu"}
            
    except Exception as e:
        print(f"Lỗi khi lưu embeddings từ file: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Lỗi: {str(e)}"}

def save_embeddings_qdrant(keyframe_dir, output_dir, collection_name="image_embeddings", model_name="OpenCLIP ViT-B-16 dfn2b", embedded_vector_file=None):
    """
    Lưu embeddings vào Qdrant từ hình ảnh hoặc từ file embeddings đã tạo sẵn
    
    Args:
        keyframe_dir: Thư mục chứa các keyframes
        output_dir: Thư mục đầu ra
        collection_name: Tên collection trong Qdrant
        model_name: Tên model sử dụng để tạo embeddings (chỉ khi tạo mới)
        embedded_vector_file: (optional) đường dẫn file JSON chứa embeddings đã tạo sẵn
        
    Returns:
        Dictionary chứa thông tin kết quả
    """
    # Nếu có file embeddings đã tạo sẵn, sử dụng hàm chuyên biệt để xử lý
    if embedded_vector_file:
        print(f"Sử dụng file embeddings đã tạo sẵn: {embedded_vector_file}")
        return save_embeddings_from_file(embedded_vector_file, keyframe_dir, output_dir, collection_name)
    
    # Trường hợp thông thường: tạo embeddings mới từ hình ảnh
    try:
        ensure_dependencies()
        
        # Đảm bảo thư mục đầu ra tồn tại
        os.makedirs(output_dir, exist_ok=True)
        
        # Phân tích model_name để lấy backbone và pretrained
        model_config = None
        
        # Thêm các cấu hình model từ config
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from backend.app.core.config import EMBEDDING_MODELS
            
            if model_name in EMBEDDING_MODELS:
                model_config = EMBEDDING_MODELS[model_name]
                print(f"Sử dụng cấu hình model từ config: {model_config}")
            else:
                print(f"Không tìm thấy cấu hình cho model {model_name} trong config")
                # Mặc định nếu không tìm thấy trong config
                if model_name == "OpenCLIP ViT-B-16 dfn2b":
                    model_config = {
                        "backbone": "ViT-B-16",
                        "pretrained": "dfn2b"
                    }
                elif model_name == "OpenCLIP ViT-B-16 SigLIP":
                    model_config = {
                        "backbone": "ViT-B-16-SigLIP",
                        "pretrained": "webli"
                    }
                else:
                    print("Sử dụng cấu hình mặc định")
                    model_config = {
                        "backbone": "ViT-B-16",
                        "pretrained": "dfn2b"
                    }
        except ImportError:
            # Fallback nếu không import được config
            print("Không thể import config, sử dụng cấu hình mặc định")
            if model_name == "OpenCLIP ViT-B-16 dfn2b":
                model_config = {
                    "backbone": "ViT-B-16",
                    "pretrained": "dfn2b"
                }
            elif model_name == "OpenCLIP ViT-B-16 SigLIP":
                model_config = {
                    "backbone": "ViT-B-16-SigLIP",
                    "pretrained": "webli"
                }
            else:
                model_config = {
                    "backbone": "ViT-B-16",
                    "pretrained": "dfn2b"
                }
        
        # Tải model OpenCLIP
        backbone = model_config["backbone"]
        pretrained = model_config["pretrained"]
        model = load_openclip_model(backbone, pretrained)
        
        if model is None:
            return {"status": "error", "message": "Không thể tải model OpenCLIP"}
        
        # Import và khởi tạo Qdrant
        from database.my_qdrant_openclip import Qdrant
        
        # Khởi tạo client
        qdrant = Qdrant(model=model)
        
        # Tạo collection nếu chưa có
        if not qdrant.is_collection_exists(collection_name):
            print(f"Tạo collection '{collection_name}'...")
            qdrant.create_collection(collection_name)
        else:
            print(f"Collection '{collection_name}' đã tồn tại.")
        
        # Tìm tất cả các ảnh trong thư mục keyframe
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_paths.extend(glob.glob(os.path.join(keyframe_dir, "**", ext), recursive=True))
        
        if not image_paths:
            return {"status": "error", "message": f"Không tìm thấy hình ảnh nào trong {keyframe_dir}"}
            
        print(f"Tìm thấy {len(image_paths)} hình ảnh")
        
        # Xử lý và tạo embeddings cho từng hình ảnh
        count = 0
        batch_size = 10  # Xử lý theo batch để tối ưu
        current_batch = []
        keyframe_mapping = {}  # Lưu mapping giữa path và id
        
        for path in tqdm(image_paths, desc="Đang tạo embeddings"):
            try:
                # Tạo random UUID cho mỗi hình ảnh
                point_id = str(uuid.uuid4())
                
                # Đọc hình ảnh
                with Image.open(path).convert("RGB") as img:
                    # Lấy tên file
                    keyframe_name = os.path.basename(path)
                    
                    # Tạo embedding bằng OpenCLIP
                    embedding = model.encode_image(img)
                    
                    # Lưu mapping
                    keyframe_mapping[point_id] = {
                        "path": path,
                        "keyframe": keyframe_name
                    }
                    
                    # Chuẩn bị dữ liệu để lưu
                    embedding_data = {
                        "id": point_id,
                        "keyframe": keyframe_name,
                        "path": path,
                        "vector": embedding
                    }
                    
                    current_batch.append(embedding_data)
                    
                    # Nếu đủ batch size thì lưu vào Qdrant
                    if len(current_batch) >= batch_size:
                        qdrant.batch_upload_points(current_batch, collection_name)
                        count += len(current_batch)
                        current_batch = []
                        print(f"Đã lưu {count}/{len(image_paths)} embeddings")
                
            except Exception as e:
                print(f"Lỗi khi xử lý hình ảnh {path}: {str(e)}")
                continue
        
        # Lưu batch cuối cùng nếu còn
        if current_batch:
            qdrant.batch_upload_points(current_batch, collection_name)
            count += len(current_batch)
            print(f"Đã lưu {count}/{len(image_paths)} embeddings")
        
        # Lưu mapping giữa ID và path
        mapping_path = os.path.join(output_dir, "keyframe_mapping.json")
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(keyframe_mapping, f, ensure_ascii=False, indent=2)
            
        print(f"Đã lưu mapping giữa ID và path vào {mapping_path}")
        
        if count > 0:
            return {
                "status": "success", 
                "message": f"Đã tạo và lưu {count} embeddings thành công vào Qdrant",
                "model": model_name,
                "mapping_file": mapping_path
            }
        else:
            return {"status": "error", "message": "Không có embedding nào được lưu"}
        
    except Exception as e:
        print(f"Lỗi khi tạo embeddings: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Lỗi: {str(e)}"}
