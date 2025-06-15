import sys
import io
import os
from models.clip import CLIP
from models.openclip import OpenCLIP
from faiss_index import FaissIndex
from app.config import *

def load_database():
    database = {}
    
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    
    database_path = os.path.abspath(DATABASE_FOLDER)
    print("\n" + "="*50)
    print(f"Đang tải chỉ mục từ thư mục: {database_path}")
    
    # Khởi tạo các models
    try:
        print("\nĐang tải CLIP Faiss...")
        try:
            clip_model = CLIP(device=DEVICE)
            clip_faiss = FaissIndex(model=clip_model)
            clip_faiss.load(os.path.join(database_path, 'clip_faiss.bin'), os.path.join(database_path, 'clip_id2path.pkl'))
            database['clip_faiss'] = clip_faiss
            print("Đã tải xong CLIP Faiss")
        except Exception as e:
            print(f"Không thể tải CLIP Faiss: {str(e)}")
            
        print("\nĐang tải OpenCLIP Faiss...")
        try:
            openclip_model = OpenCLIP(backbone='ViT-B-32', pretrained='laion2b_s34b_b79k', device=DEVICE)
            openclip_faiss = FaissIndex(model=openclip_model)
            openclip_faiss.load(os.path.join(database_path, 'openclip_faiss.bin'), os.path.join(database_path, 'openclip_id2path.pkl'))
            database['openclip_faiss'] = openclip_faiss
            print("Đã tải xong OpenCLIP Faiss")
        except Exception as e:
            print(f"Không thể tải OpenCLIP Faiss: {str(e)}")
    
    except Exception as e:
        import traceback
        print(f"Lỗi khi khởi tạo database: {str(e)}\n{traceback.format_exc()}")
    
    return database