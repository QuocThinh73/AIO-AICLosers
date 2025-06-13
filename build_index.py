import os
import sys
import glob
import torch
from PIL import Image

# Thiết lập encoding cho đầu ra console
sys.stdout.reconfigure(encoding='utf-8')

try:
    from models.clip import CLIP
    from models.openclip import OpenCLIP
    from faiss_index import FaissIndex
except ImportError as e:
    print(f"Lỗi import: {e}")
    print("Vui lòng đảm bảo đã cài đặt đầy đủ các thư viện cần thiết.")
    sys.exit(1)

# Cấu hình
IMAGE_FOLDER = "data/keyframes"  # Thư mục chứa các ảnh cần đánh chỉ mục
OUTPUT_DIR = "database"

# Tạo thư mục đầu ra nếu chưa tồn tại
os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_clip_index():
    print("Đang xây dựng chỉ mục CLIP...")
    clip_model = CLIP(device="cuda" if torch.cuda.is_available() else "cpu")
    faiss_index = FaissIndex(clip_model)
    
    # Lấy danh sách các file ảnh
    image_paths = glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")) + \
                 glob.glob(os.path.join(IMAGE_FOLDER, "*.png"))
    
    if not image_paths:
        print(f"Không tìm thấy ảnh trong thư mục {IMAGE_FOLDER}")
        return
    
    print(f"Tìm thấy {len(image_paths)} ảnh để đánh chỉ mục...")
    
    # Xây dựng chỉ mục
    faiss_index.build(image_paths, "clip", output_dir=OUTPUT_DIR)
    print("Đã xây dựng xong chỉ mục CLIP!")

def build_openclip_index():
    print("Đang xây dựng chỉ mục OpenCLIP...")
    openclip_model = OpenCLIP(
        backbone="ViT-B-32", 
        pretrained="laion2b_s34b_b79k",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    faiss_index = FaissIndex(openclip_model)
    
    # Lấy danh sách các file ảnh
    image_paths = glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")) + \
                 glob.glob(os.path.join(IMAGE_FOLDER, "*.png"))
    
    if not image_paths:
        print(f"Không tìm thấy ảnh trong thư mục {IMAGE_FOLDER}")
        return
    
    print(f"Tìm thấy {len(image_paths)} ảnh để đánh chỉ mục...")
    
    # Xây dựng chỉ mục
    faiss_index.build(image_paths, "openclip", output_dir=OUTPUT_DIR)
    print("Đã xây dựng xong chỉ mục OpenCLIP!")

if __name__ == "__main__":
    # Kiểm tra xem có GPU không
    device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"[INFO] Dang su dung: {device}")
    
    try:
        # Xây dựng các chỉ mục
        print("\n[INFO] Bat dau xay dung chi muc CLIP...")
        build_clip_index()
        
        print("\n[INFO] Bat dau xay dung chi muc OpenCLIP...")
        build_openclip_index()
        
        print("\n[SUCCESS] Da hoan thanh xay dung tat ca chi muc!")
        print(f"Cac file chi muc da duoc luu tai thu muc: {os.path.abspath(OUTPUT_DIR)}")
    except Exception as e:
        print(f"\n[ERROR] Co loi xay ra: {str(e)}")
        print("Vui long kiem tra lai duong dan den thu muc anh hoocac thu vien da cai dat.")
