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
def create_index(es, index_name="ocr_results"):
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
                "text_results": {
                    "type": "nested",
                    "properties": {
                        "text": {"type": "text"},
                        "confidence": {"type": "float"},
                        "box": {
                            "properties": {
                                "top_left": {
                                    "properties": {
                                        "x": {"type": "float"},
                                        "y": {"type": "float"}
                                    }
                                },
                                "top_right": {
                                    "properties": {
                                        "x": {"type": "float"},
                                        "y": {"type": "float"}
                                    }
                                },
                                "bottom_right": {
                                    "properties": {
                                        "x": {"type": "float"},
                                        "y": {"type": "float"}
                                    }
                                },
                                "bottom_left": {
                                    "properties": {
                                        "x": {"type": "float"},
                                        "y": {"type": "float"}
                                    }
                                }
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

# Lưu kết quả OCR vào Elasticsearch với đường dẫn đúng
def index_ocr_results(es, ocr_dir, index_name="ocr_results"):
    # Lấy danh sách các file JSON chứa kết quả OCR
    json_files = glob.glob(os.path.join(ocr_dir, "**/*.json"), recursive=True)
    print(f"Tìm thấy {len(json_files)} file kết quả OCR")
    
    total_indexed = 0
    
    for json_file in tqdm(json_files, desc="Đang lưu vào Elasticsearch"):
        try:
            print(f"Đang xử lý file: {json_file}")
            with open(json_file, "r", encoding="utf-8") as f:
                ocr_results = json.load(f)
            
            # Lấy tên video từ tên file
            # Ví dụ: L01/V001.json -> L01_V001
            video_file = os.path.basename(json_file)
            video_name = video_file.replace(".json", "")
            lesson_name = os.path.basename(os.path.dirname(json_file))
            if lesson_name.startswith("L"):
                video_name = f"{lesson_name}_{video_name}"
            
            print(f"Tên file: {video_file}, video_name sau xử lý: {video_name}")
            
            # Xử lý từng keyframe trong file
            for entry in ocr_results:
                keyframe = entry.get("image", "")
                text_results = entry.get("results", [])
                
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
                
                # Format OCR text results
                formatted_results = []
                for result in text_results:
                    text = result.get("text", "")
                    confidence = result.get("confidence", 0.0)
                    box = result.get("box", [])
                    
                    # Kiểm tra nếu có đủ 4 điểm trong box
                    if len(box) == 4:
                        formatted_result = {
                            "text": text,
                            "confidence": confidence,
                            "box": {
                                "top_left": {"x": box[0][0], "y": box[0][1]},
                                "top_right": {"x": box[1][0], "y": box[1][1]},
                                "bottom_right": {"x": box[2][0], "y": box[2][1]},
                                "bottom_left": {"x": box[3][0], "y": box[3][1]}
                            }
                        }
                        formatted_results.append(formatted_result)
                
                # Tạo document để lưu vào Elasticsearch
                doc = {
                    "video_name": video_name,
                    "keyframe": keyframe,
                    "keyframe_path": correct_path,
                    "text_results": formatted_results
                }
                
                # Lưu vào Elasticsearch với ID duy nhất
                doc_id = f"{video_name}_{keyframe}"
                es.index(index=index_name, id=doc_id, body=doc)
                total_indexed += 1
                
        except Exception as e:
            print(f"Lỗi khi xử lý file {json_file}: {str(e)}")
    
    print(f"Đã lưu thành công {total_indexed} keyframes vào Elasticsearch")

# Hàm trợ giúp để cài đặt thư viện nếu cần
def ensure_dependencies():
    try:
        import elasticsearch
    except ImportError:
        print("Thư viện elasticsearch chưa được cài đặt. Đang cài đặt...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "elasticsearch"])
        print("Đã cài đặt elasticsearch")

def main():
    # Đảm bảo đã cài đặt các thư viện cần thiết
    try:
        ensure_dependencies()
    except Exception as e:
        print(f"Lỗi khi cài đặt thư viện: {str(e)}")
    
    es = connect_to_elasticsearch()
    if es is None:
        return
    
    # Tạo index mới
    create_index(es)
    
    # Thư mục chứa các file JSON kết quả OCR
    ocr_dir = os.path.abspath("./database/ocr")
    print(f"Đang tìm các file OCR trong thư mục: {ocr_dir}")
    
    # Kiểm tra thư mục có tồn tại không
    if not os.path.exists(ocr_dir):
        print(f"Thư mục không tồn tại: {ocr_dir}")
        # Thử đường dẫn khác
        alt_paths = [
            os.path.abspath("../database/ocr"),
            "database/ocr"
        ]
        
        for path in alt_paths:
            print(f"Đang thử đường dẫn thay thế: {path}")
            if os.path.exists(path):
                ocr_dir = path
                print(f"Đã tìm thấy thư mục: {ocr_dir}")
                break
    
    # Lưu dữ liệu vào Elasticsearch với đường dẫn đúng
    index_ocr_results(es, ocr_dir)
    
if __name__ == "__main__":
    import sys
    main()
