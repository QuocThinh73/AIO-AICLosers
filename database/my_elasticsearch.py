import os
import json
import sys
import glob
from tqdm import tqdm
from elasticsearch import Elasticsearch


class MyElasticsearch:
    def __init__(self, host="http://localhost:9200"):
        """
        Khởi tạo lớp MyElasticsearch
        
        Args:
            host (str): URL của Elasticsearch server
        """
        self.host = host
        self.es = None
        
    def connect(self):
        """
        Kết nối đến Elasticsearch server
        
        Returns:
            bool: True nếu kết nối thành công, False nếu không
        """
        try:
            self.es = Elasticsearch([self.host])
            if self.es.ping():
                print("Đã kết nối thành công với Elasticsearch")
                return True
            else:
                print("Không thể kết nối đến Elasticsearch. Vui lòng kiểm tra lại")
                return False
        except Exception as e:
            print(f"Lỗi khi kết nối đến Elasticsearch: {str(e)}")
            return False
    
    def index_exists(self, index_name):
        """
        Kiểm tra xem index có tồn tại không
        
        Args:
            index_name (str): Tên của index
            
        Returns:
            bool: True nếu index tồn tại, False nếu không
        """
        return self.es.indices.exists(index=index_name)
    
    def delete_index(self, index_name):
        """
        Xóa index nếu tồn tại
        
        Args:
            index_name (str): Tên của index
            
        Returns:
            bool: True nếu xóa thành công, False nếu không
        """
        if self.index_exists(index_name):
            self.es.indices.delete(index=index_name)
            print(f"Đã xóa index {index_name}")
            return True
        return False
    
    def create_detection_index(self, index_name="groundingdino"):
        """
        Tạo index cho dữ liệu detection với mapping phù hợp
        
        Args:
            index_name (str): Tên của index
        """
        if self.index_exists(index_name):
            print(f"Đang xóa index {index_name} cũ...")
            self.delete_index(index_name)
        
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
        
        self.es.indices.create(index=index_name, body=mapping)
        print(f"Đã tạo index {index_name} với mapping cho detection")
    
    def create_ocr_index(self, index_name="ocr_results"):
        """
        Tạo index cho dữ liệu OCR với mapping phù hợp
        
        Args:
            index_name (str): Tên của index
        """
        if self.index_exists(index_name):
            print(f"Đang xóa index {index_name} cũ...")
            self.delete_index(index_name)
        
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
        
        self.es.indices.create(index=index_name, body=mapping)
        print(f"Đã tạo index {index_name} với mapping cho OCR")
    
    def index_document(self, index_name, doc_id, document):
        """
        Lưu document vào index
        
        Args:
            index_name (str): Tên của index
            doc_id (str): ID của document
            document (dict): Document cần lưu
            
        Returns:
            dict: Kết quả của việc lưu document
        """
        try:
            return self.es.index(index=index_name, id=doc_id, body=document)
        except Exception as e:
            print(f"Lỗi khi lưu document: {str(e)}")
            return None
    
    def format_detection_data(self, video_name, keyframe, caption, objects):
        """
        Format dữ liệu detection để lưu vào Elasticsearch
        
        Args:
            video_name (str): Tên của video
            keyframe (str): Tên của keyframe
            caption (str): Caption của keyframe
            objects (list): Danh sách các object được phát hiện
            
        Returns:
            tuple: (doc_id, document) - ID và document đã được format
        """
        if "_V" in video_name:
            video_parts = video_name.split("_V")
            if len(video_parts) == 2:
                L_part = video_parts[0]  # L01
                V_part = f"V{video_parts[1]}"  # V001
                keyframe_path = f"{L_part}/{V_part}/{keyframe}"
            else:
                keyframe_path = f"{video_name}/{keyframe}"
        else:
            keyframe_path = f"{video_name}/{keyframe}"
        
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
            "keyframe_path": keyframe_path,
            "caption": caption,
            "objects": formatted_objects
        }
        
        doc_id = f"{video_name}_{keyframe}"
        return doc_id, doc
    
    def format_ocr_data(self, video_name, keyframe, text_results):
        """
        Format dữ liệu OCR để lưu vào Elasticsearch
        
        Args:
            video_name (str): Tên của video
            keyframe (str): Tên của keyframe
            text_results (list): Danh sách các kết quả OCR
            
        Returns:
            tuple: (doc_id, document) - ID và document đã được format
        """
        if "_V" in video_name:
            video_parts = video_name.split("_V")
            if len(video_parts) == 2:
                L_part = video_parts[0]  # L01
                V_part = f"V{video_parts[1]}"  # V001
                keyframe_path = f"{L_part}/{V_part}/{keyframe}"
            else:
                keyframe_path = f"{video_name}/{keyframe}"
        else:
            keyframe_path = f"{video_name}/{keyframe}"
        
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
            "keyframe_path": keyframe_path,
            "text_results": formatted_results
        }
        
        doc_id = f"{video_name}_{keyframe}"
        return doc_id, doc
    
    def index_detection_results(self, detection_dir, index_name="groundingdino"):
        """
        Lưu kết quả detection vào Elasticsearch
        
        Args:
            detection_dir (str): Đường dẫn đến thư mục chứa kết quả detection
            index_name (str): Tên của index
            
        Returns:
            dict: Trạng thái của việc lưu kết quả
        """
        try:
            if not self.es:
                if not self.connect():
                    return {"status": "error", "message": "Không thể kết nối đến Elasticsearch"}
            
            self.create_detection_index(index_name)
            
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
                        
                        doc_id, doc = self.format_detection_data(video_name, keyframe, caption, objects)
                        self.index_document(index_name, doc_id, doc)
                        total_indexed += 1
                        
                except Exception as e:
                    print(f"Lỗi khi xử lý file {json_file}: {str(e)}")
            
            if total_indexed > 0:
                return {"status": "success", "message": f"Đã lưu thành công {total_indexed} keyframes vào Elasticsearch"}
            else:
                return {"status": "error", "message": "Không có dữ liệu nào được lưu vào Elasticsearch"}
        
        except Exception as e:
            return {"status": "error", "message": f"Lỗi: {str(e)}"}
    
    def index_ocr_results(self, ocr_dir, index_name="ocr_results"):
        """
        Lưu kết quả OCR vào Elasticsearch
        
        Args:
            ocr_dir (str): Đường dẫn đến thư mục chứa kết quả OCR
            index_name (str): Tên của index
            
        Returns:
            dict: Trạng thái của việc lưu kết quả
        """
        try:
            if not self.es:
                if not self.connect():
                    return {"status": "error", "message": "Không thể kết nối đến Elasticsearch"}
            
            self.create_ocr_index(index_name)
            
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
                        
                        doc_id, doc = self.format_ocr_data(video_name, keyframe, text_results)
                        self.index_document(index_name, doc_id, doc)
                        total_indexed += 1
                        
                except Exception as e:
                    print(f"Lỗi khi xử lý file {json_file}: {str(e)}")
            
            if total_indexed > 0:
                return {"status": "success", "message": f"Đã lưu thành công {total_indexed} keyframes vào Elasticsearch"}
            else:
                return {"status": "error", "message": "Không có dữ liệu nào được lưu vào Elasticsearch"}
        
        except Exception as e:
            return {"status": "error", "message": f"Lỗi: {str(e)}"}
    
    def search_by_text(self, query, index_name, field="caption", size=10):
        """
        Tìm kiếm theo text trong Elasticsearch
        
        Args:
            query (str): Query cần tìm kiếm
            index_name (str): Tên của index
            field (str): Trường cần tìm kiếm
            size (int): Số lượng kết quả tối đa
            
        Returns:
            list: Danh sách các kết quả tìm kiếm
        """
        try:
            search_query = {
                "size": size,
                "query": {
                    "match": {
                        field: query
                    }
                }
            }
            
            response = self.es.search(index=index_name, body=search_query)
            return response["hits"]["hits"]
        except Exception as e:
            print(f"Lỗi khi tìm kiếm: {str(e)}")
            return []
    
    def search_by_video_name(self, video_name, index_name, size=100):
        """
        Tìm kiếm theo tên video trong Elasticsearch
        
        Args:
            video_name (str): Tên video cần tìm
            index_name (str): Tên của index
            size (int): Số lượng kết quả tối đa
            
        Returns:
            list: Danh sách các kết quả tìm kiếm
        """
        try:
            search_query = {
                "size": size,
                "query": {
                    "term": {
                        "video_name.keyword": video_name
                    }
                }
            }
            
            response = self.es.search(index=index_name, body=search_query)
            return response["hits"]["hits"]
        except Exception as e:
            print(f"Lỗi khi tìm kiếm: {str(e)}")
            return []
    
    def search_nested_objects(self, object_name, index_name="groundingdino", size=10):
        """
        Tìm kiếm theo tên đối tượng trong các object được phát hiện
        
        Args:
            object_name (str): Tên đối tượng cần tìm
            index_name (str): Tên của index
            size (int): Số lượng kết quả tối đa
            
        Returns:
            list: Danh sách các kết quả tìm kiếm
        """
        try:
            search_query = {
                "size": size,
                "query": {
                    "nested": {
                        "path": "objects",
                        "query": {
                            "match": {
                                "objects.object": object_name
                            }
                        }
                    }
                }
            }
            
            response = self.es.search(index=index_name, body=search_query)
            return response["hits"]["hits"]
        except Exception as e:
            print(f"Lỗi khi tìm kiếm: {str(e)}")
            return []
    
    def search_nested_text(self, text, index_name="ocr_results", size=10):
        """
        Tìm kiếm theo text trong kết quả OCR
        
        Args:
            text (str): Text cần tìm
            index_name (str): Tên của index
            size (int): Số lượng kết quả tối đa
            
        Returns:
            list: Danh sách các kết quả tìm kiếm
        """
        try:
            search_query = {
                "size": size,
                "query": {
                    "nested": {
                        "path": "text_results",
                        "query": {
                            "match": {
                                "text_results.text": text
                            }
                        }
                    }
                }
            }
            
            response = self.es.search(index=index_name, body=search_query)
            return response["hits"]["hits"]
        except Exception as e:
            print(f"Lỗi khi tìm kiếm: {str(e)}")
            return []


def ensure_elasticsearch_dependencies():
    """
    Đảm bảo thư viện elasticsearch đã được cài đặt
    """
    try:
        import elasticsearch
    except ImportError:
        print("Thư viện elasticsearch chưa được cài đặt. Đang cài đặt...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "elasticsearch"])
        print("Đã cài đặt elasticsearch")
