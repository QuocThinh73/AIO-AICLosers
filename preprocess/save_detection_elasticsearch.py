import os
import json
import glob
import sys
from tqdm import tqdm
from database.my_elasticsearch import MyElasticsearch, ensure_elasticsearch_dependencies

def index_detection_results(detection_dir, index_name="groundingdino"):
    try:
        ensure_elasticsearch_dependencies()
        
        es_client = MyElasticsearch()
        if not es_client.connect():
            return {"status": "error", "message": "Không thể kết nối đến Elasticsearch"}
        
        result = es_client.index_detection_results(detection_dir, index_name)
        return result
        
    except Exception as e:
        return {"status": "error", "message": f"Lỗi: {str(e)}"}
