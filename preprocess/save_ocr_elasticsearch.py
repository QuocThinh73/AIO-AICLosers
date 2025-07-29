import os
import json
from elasticsearch import Elasticsearch
import glob
from tqdm import tqdm

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

def create_index(es, index_name="ocr_results"):
    if es.indices.exists(index=index_name):
        print(f"Đang xóa index {index_name} cũ...")
        es.indices.delete(index=index_name)
    
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
    
    es.indices.create(index=index_name, body=mapping)
    print(f"Đã tạo index {index_name} với mapping")

def index_ocr_results(ocr_dir, index_name="ocr_results"):
    json_files = glob.glob(os.path.join(ocr_dir, "**/*.json"), recursive=True)
    print(f"Tìm thấy {len(json_files)} file kết quả OCR")
    
    total_indexed = 0
    
    for json_file in tqdm(json_files, desc="Đang lưu vào Elasticsearch"):
        try:
            print(f"Đang xử lý file: {json_file}")
            with open(json_file, "r", encoding="utf-8") as f:
                ocr_results = json.load(f)
            
            video_file = os.path.basename(json_file)
            video_name = video_file.replace(".json", "")
            lesson_name = os.path.basename(os.path.dirname(json_file))
            if lesson_name.startswith("L"):
                video_name = f"{lesson_name}_{video_name}"
            
            print(f"Tên file: {video_file}, video_name sau xử lý: {video_name}")
            
            for entry in ocr_results:
                keyframe = entry.get("image", "")
                text_results = entry.get("results", [])
                
               
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
                
                
                formatted_results = []
                for result in text_results:
                    text = result.get("text", "")
                    confidence = result.get("confidence", 0.0)
                    box = result.get("box", [])
                    
                        
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
                
                    
                doc = {
                    "video_name": video_name,
                    "keyframe": keyframe,
                    "keyframe_path": correct_path,
                    "text_results": formatted_results
                }
                
                
                doc_id = f"{video_name}_{keyframe}"
                es.index(index=index_name, id=doc_id, body=doc)
                total_indexed += 1
                
        except Exception as e:
            print(f"Lỗi khi xử lý file {json_file}: {str(e)}")
    
    print(f"Đã lưu thành công {total_indexed} keyframes vào Elasticsearch")

def ensure_dependencies():
    try:
        import elasticsearch
    except ImportError:
        print("Thư viện elasticsearch chưa được cài đặt. Đang cài đặt...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "elasticsearch"])
        print("Đã cài đặt elasticsearch")


def index_ocr_results(ocr_dir, index_name="ocr_results"):
    try:
        ensure_dependencies()
        
        es = connect_to_elasticsearch()
        if es is None:
            return {"status": "error", "message": "Không thể kết nối đến Elasticsearch"}
        
        
        create_index(es, index_name)
        
        if not os.path.exists(ocr_dir):
            return {"status": "error", "message": f"Thư mục không tồn tại: {ocr_dir}"}
        
        json_files = glob.glob(os.path.join(ocr_dir, "**/*.json"), recursive=True)
        print(f"Tìm thấy {len(json_files)} file kết quả OCR")
        
        if len(json_files) == 0:
            return {"status": "error", "message": f"Không tìm thấy file JSON nào trong {ocr_dir}"}
        
        total_indexed = 0
        
        for json_file in tqdm(json_files, desc="Đang lưu vào Elasticsearch"):
            try:
                print(f"Đang xử lý file: {json_file}")
                with open(json_file, "r", encoding="utf-8") as f:
                    ocr_results = json.load(f)
                
                video_file = os.path.basename(json_file)
                video_name = video_file.replace("_ocr.json", "")
                print(f"Tên file: {video_file}, video_name sau xử lý: {video_name}")
                
                for entry in ocr_results:
                    keyframe = entry.get("image", "")
                    text_results = entry.get("results", [])
                    
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
                    
                    doc = {
                        "video_name": video_name,
                        "keyframe": keyframe,
                        "keyframe_path": correct_path,
                        "text_results": formatted_results
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

