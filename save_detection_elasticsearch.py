import os
import json
from elasticsearch import Elasticsearch
import glob
from tqdm import tqdm

# Kết nối đến Elasticsearch
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

# Tạo index nếu chưa tồn tại
def create_index(es, index_name="groundingdino"):
    if es.indices.exists(index=index_name):
        print(f"Đang xóa index {index_name} cũ...")
        es.indices.delete(index=index_name)
    
    # Định nghĩa mapping cho index
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
    
    # Tạo index với mapping
    es.indices.create(index=index_name, body=mapping)
    print(f"Đã tạo index {index_name} với mapping")

# Lưu kết quả detection vào Elasticsearch với đường dẫn đúng
def index_detection_results(es, detection_dir, index_name="groundingdino"):
    # Lấy danh sách các file JSON chứa kết quả detection
    json_files = glob.glob(os.path.join(detection_dir, "*.json"))
    print(f"Tìm thấy {len(json_files)} file kết quả detection")
    
    total_indexed = 0
    
    for json_file in tqdm(json_files, desc="Đang lưu vào Elasticsearch"):
        try:
            print(f"Đang xử lý file: {json_file}")
            with open(json_file, "r", encoding="utf-8") as f:
                detections = json.load(f)
            
            # Lấy tên video từ tên file và loại bỏ hậu tố _detection
            # Ví dụ: L01_V001_detection.json -> L01_V001
            video_file = os.path.basename(json_file)
            video_name = video_file.replace("_detection.json", "")
            print(f"Tên file: {video_file}, video_name sau xử lý: {video_name}")
            
            # Xử lý từng keyframe trong file
            for entry in detections:
                keyframe = entry.get("keyframe", "")
                caption = entry.get("caption", "")
                objects = entry.get("objects", [])
                
                # Tạo đường dẫn đúng theo cấu trúc thư mục thực tế L01/V001/L01_V001_000000.jpg
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
                
                # Format objects (phần còn lại giữ nguyên)
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
                
                # Tạo document để lưu vào Elasticsearch
                doc = {
                    "video_name": video_name,
                    "keyframe": keyframe,
                    "keyframe_path": correct_path,
                    "caption": caption,
                    "objects": formatted_objects
                }
                
                # Lưu vào Elasticsearch với ID duy nhất
                doc_id = f"{video_name}_{keyframe}"
                es.index(index=index_name, id=doc_id, body=doc)
                total_indexed += 1
                
        except Exception as e:
            print(f"Lỗi khi xử lý file {json_file}: {str(e)}")
    
    print(f"Đã lưu thành công {total_indexed} keyframes vào Elasticsearch")

def main():
    es = connect_to_elasticsearch()
    if es is None:
        return
    
    # Tạo index mới
    create_index(es)
    
    # Thư mục chứa các file JSON kết quả detection - cập nhật đúng đường dẫn
    detection_dir = os.path.abspath("./database/detection_results")
    print(f"Đang tìm các file detection trong thư mục: {detection_dir}")
    
    # Kiểm tra thư mục có tồn tại không
    if not os.path.exists(detection_dir):
        print(f"Thư mục không tồn tại: {detection_dir}")
        # Thử đường dẫn khác
        alt_paths = [
            os.path.abspath("../database/detection_results"),
            "f:/1code/1Research/CaptionandDetection/flask_app/database/detection_results"
        ]
        
        for path in alt_paths:
            print(f"Đang thử đường dẫn thay thế: {path}")
            if os.path.exists(path):
                detection_dir = path
                print(f"Đã tìm thấy thư mục: {detection_dir}")
                break
    
    # Lưu dữ liệu vào Elasticsearch với đường dẫn đúng
    index_detection_results(es, detection_dir)
    
if __name__ == "__main__":
    main()