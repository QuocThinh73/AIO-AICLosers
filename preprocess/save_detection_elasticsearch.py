import os
import json
import glob
import sys
from tqdm import tqdm
from elasticsearch import Elasticsearch


def connect_to_elasticsearch():
    try:
        es = Elasticsearch(["http://localhost:9200"])
        if es.ping():
            print("Đã kết nối thành công với Elasticsearch")
            return es
        else:
            print("Không thể kết nối đến Elasticsearch. Vui lòng kiểm tra lại")
            return None
    except Exception as e:
        print(f"Lỗi khi kết nối đến Elasticsearch: {str(e)}")
        return None

def create_index(es, index_name="groundingdino"):
    if es.indices.exists(index=index_name):
        print(f"Đang xóa index {index_name} cũ...")
        es.indices.delete(index=index_name)
    
    mapping = {
        "mappings": {
            "properties": {
                "video_name": {"type": "keyword"},
                "keyframe": {"type": "keyword"},
                "keyframe_path": {"type": "keyword"},
                "caption": {"type": "text"},
                "objects": {
                    "type": "nested",
                    "properties": {
                        "prompt": {"type": "text"},
                        "object": {"type": "keyword"},
                        "score": {"type": "float"},
                        "box": {
                            "properties": {
                                "x1": {"type": "float"},
                                "y1": {"type": "float"},
                                "x2": {"type": "float"},
                                "y2": {"type": "float"}
                            }
                        }
                    }
                }
            }
        }
    }
    
    es.indices.create(index=index_name, body=mapping)
    print(f"Đã tạo index {index_name} với mapping")

def ensure_dependencies():
    try:
        import elasticsearch
    except ImportError:
        print("Thư viện elasticsearch chưa được cài đặt. Đang cài đặt...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "elasticsearch"])
        print("Đã cài đặt elasticsearch")

def index_detection_results(detection_dir, index_name="groundingdino"):
    try:
        ensure_dependencies()
        
        es = connect_to_elasticsearch()
        if es is None:
            return {"status": "error", "message": "Không thể kết nối đến Elasticsearch"}
        
        create_index(es, index_name)
        
        if not os.path.exists(detection_dir):
            return {"status": "error", "message": f"Thư mục không tồn tại: {detection_dir}"}
        
        json_files = glob.glob(os.path.join(detection_dir, "*.json"))
        print(f"Tìm thấy {len(json_files)} file kết quả detection")
        
        if len(json_files) == 0:
            return {"status": "error", "message": f"Không tìm thấy file JSON nào trong {detection_dir}"}
        
        total_indexed = 0
        
        for json_file in tqdm(json_files, desc="Đang lưu vào Elasticsearch"):
            try:
                print(f"Đang xử lý file: {json_file}")
                with open(json_file, "r", encoding="utf-8") as f:
                    detections = json.load(f)
                
                video_file = os.path.basename(json_file)
                video_name = video_file.replace("_detection.json", "")
                print(f"Tên file: {video_file}, video_name sau xử lý: {video_name}")
                for entry in detections:
                    keyframe = entry.get("keyframe", "")
                    caption = entry.get("caption", "")
                    objects = entry.get("objects", [])
                    
                    if "_V" in video_name:
                        video_parts = video_name.split("_V")
                        if len(video_parts) == 2:
                            L_part = video_parts[0]  # L01
                            V_part = f"V{video_parts[1]}"  # V001
                            correct_path = f"{L_part}/{V_part}/{keyframe}"
                            print(f"Đường dẫn keyframe đã tạo: {correct_path}")
                        else:
                            correct_path = f"{video_name}/{keyframe}"
                    else:
                        correct_path = f"{video_name}/{keyframe}"
                    
                    # Format objects
                    formatted_objects = []
                    for obj in objects:
                        box_array = obj["box"]
                        formatted_object = {
                            "prompt": obj["prompt"],
                            "object": obj["object"],
                            "score": obj["score"],
                            "box": {
                                "x1": box_array[0],
                                "y1": box_array[1],
                                "x2": box_array[2],
                                "y2": box_array[3]
                            }
                        }
                        formatted_objects.append(formatted_object)
                    
                    doc = {
                        "video_name": video_name,
                        "keyframe": keyframe,
                        "keyframe_path": correct_path,
                        "caption": caption,
                        "objects": formatted_objects
                    }
                    
                    doc_id = f"{video_name}_{keyframe}"
                    es.index(index=index_name, id=doc_id, body=doc)
                    total_indexed += 1
                    
            except Exception as e:
                print(f"Lỗi khi xử lý file {json_file}: {str(e)}")
        
        if total_indexed > 0:
            return {"status": "success", "message": f"Đã lưu thành công {total_indexed} keyframes vào Elasticsearch"}
        else:
            return {"status": "error", "message": "Không có dữ liệu nào được lưu vào Elasticsearch"}
    
    except Exception as e:
        return {"status": "error", "message": f"Lỗi: {str(e)}"}
