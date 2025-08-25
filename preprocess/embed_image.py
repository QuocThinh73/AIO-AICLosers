import os
import json
import tqdm
import sys
import subprocess
from PIL import Image


def _ensure_dependencies():
    """Đảm bảo các thư viện phụ thuộc đã được cài đặt"""
    print("Kiểm tra và cài đặt các thư viện cần thiết...")
    
    # Kiểm tra xem có đang chạy trong Kaggle không
    in_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    
    required_packages = [
        "open_clip_torch",
        "ftfy",
        "regex"
    ]
    
    for package in required_packages:
        try:
            # Thử import package
            if package == "open_clip_torch":
                try:
                    import open_clip
                    print(f"Đã tìm thấy {package}")
                    continue
                except ImportError:
                    pass
            
            print(f"Đang cài đặt {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
            print(f"Đã cài đặt {package} thành công")
                
        except Exception as e:
            print(f"Cảnh báo: Không thể cài đặt {package}. Lỗi: {e}")
            # Tiếp tục vì package có thể đã được cài đặt

# Đảm bảo các thư viện phụ thuộc đã được cài đặt
_ensure_dependencies()


def get_image_embedder(backbone, pretrained):
    from models.openclip import OpenCLIP
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_embedder = OpenCLIP(backbone, pretrained, device=device)
    return image_embedder

def process_video(keyframe_dir, output_embedded_vector_path, image_embedder):
    embedded_vectors = []
    # Chỉ xử lý các file ảnh
    image_files = [f for f in sorted(os.listdir(keyframe_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"Không tìm thấy file ảnh trong {keyframe_dir}")
        return 0
    
    print(f"Tìm thấy {len(image_files)} ảnh trong {keyframe_dir}")
        
    # Sử dụng tqdm để hiển thị thanh tiến trình
    for keyframe_name in tqdm.tqdm(image_files, desc="Embedding images"):
        keyframe_path = os.path.join(keyframe_dir, keyframe_name)
        image = Image.open(keyframe_path).convert("RGB")
        embedded_vector = image_embedder.encode_image(image)

        embedded_vectors.append({
            "keyframe": keyframe_name,
            "embedded_vector": embedded_vector.tolist()
        })

    with open(output_embedded_vector_path, "w", encoding="utf-8") as f:
        json.dump(embedded_vectors, f)
        
    return len(embedded_vectors)
    

def embed_image(input_keyframe_dir, output_embedded_vector_dir, mode, backbone, pretrained, lesson_name=None, video_name=None):
    """
    Tạo embeddings cho các keyframes sử dụng OpenCLIP model mà không cần file mapping.json

    Args:
        input_keyframe_dir: Thư mục chứa keyframes
        output_embedded_vector_dir: Thư mục đầu ra để lưu các file embeddings
        mode: Chế độ xử lý ('all', 'lesson', 'video')
        backbone: Model backbone (ví dụ: 'ViT-B-16')
        pretrained: Pretrained weights (ví dụ: 'dfn2b' hoặc 'webli')
        lesson_name: Tên bài học khi mode là 'lesson' hoặc 'video'
        video_name: Tên video khi mode là 'video'
    
    Returns:
        Dict: Thông tin về quá trình xử lý
    """
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_embedded_vector_dir, exist_ok=True)
    print(f"Thư mục đầu ra: {output_embedded_vector_dir}")

    # Khởi tạo image embedder
    print(f"Khởi tạo image embedder với backbone={backbone}, pretrained={pretrained}")
    image_embedder = get_image_embedder(backbone, pretrained)
    
    # Kết quả xử lý
    results = {
        "mode": mode,
        "input_dir": input_keyframe_dir,
        "output_dir": output_embedded_vector_dir,
        "processed_videos": []
    }

    # Xử lý theo mode
    if mode == "all":
        # Xử lý tất cả các bài học
        lessons = [d for d in os.listdir(input_keyframe_dir) 
                if os.path.isdir(os.path.join(input_keyframe_dir, d))]
        
        print(f"Tìm thấy {len(lessons)} bài học")
        
        total_processed = 0
        for lesson in lessons:
            lesson_dir = os.path.join(input_keyframe_dir, lesson)
            lesson_output_dir = os.path.join(output_embedded_vector_dir, lesson)
            os.makedirs(lesson_output_dir, exist_ok=True)
            
            # Xử lý tất cả video trong bài học
            videos = [d for d in os.listdir(lesson_dir) 
                    if os.path.isdir(os.path.join(lesson_dir, d))]
            
            if not videos:
                print(f"Không tìm thấy video nào trong bài học {lesson}")
                continue
                
            print(f"Bài học {lesson}: tìm thấy {len(videos)} videos")
            
            for video in videos:
                video_dir = os.path.join(lesson_dir, video)
                output_path = os.path.join(lesson_output_dir, f"{video}_embedded_vector.json")
                
                print(f"Đang xử lý {lesson}/{video}...")
                num_processed = process_video(video_dir, output_path, image_embedder)
                
                if num_processed:
                    total_processed += 1
                    results["processed_videos"].append({
                        "lesson": lesson,
                        "video": video,
                        "keyframes": num_processed,
                        "output_file": output_path
                    })
        
        results["status"] = "success"
        results["message"] = f"Đã xử lý {total_processed} videos từ {len(lessons)} bài học"
        results["total_videos"] = total_processed
    
    elif mode == "lesson":
        # Xử lý một bài học cụ thể
        lesson_dir = os.path.join(input_keyframe_dir, lesson_name)
        lesson_output_dir = os.path.join(output_embedded_vector_dir, lesson_name)
        os.makedirs(lesson_output_dir, exist_ok=True)
        
        videos = [d for d in os.listdir(lesson_dir) 
                if os.path.isdir(os.path.join(lesson_dir, d))]
        
        print(f"Bài học {lesson_name}: tìm thấy {len(videos)} videos")
        
        total_processed = 0
        for video in videos:
            video_dir = os.path.join(lesson_dir, video)
            output_path = os.path.join(lesson_output_dir, f"{video}_embedded_vector.json")
            
            print(f"Đang xử lý {lesson_name}/{video}...")
            num_processed = process_video(video_dir, output_path, image_embedder)
            
            if num_processed:
                total_processed += 1
                results["processed_videos"].append({
                    "lesson": lesson_name,
                    "video": video,
                    "keyframes": num_processed,
                    "output_file": output_path
                })
        
        results["status"] = "success"
        results["message"] = f"Đã xử lý {total_processed} videos từ bài học {lesson_name}"
        results["total_videos"] = total_processed
    
    elif mode == "video":
        # Xử lý một video cụ thể
        video_dir = os.path.join(input_keyframe_dir, lesson_name, video_name)
        
        # Tạo thư mục output cho lesson này
        lesson_output_dir = os.path.join(output_embedded_vector_dir, lesson_name)
        os.makedirs(lesson_output_dir, exist_ok=True)
        
        output_path = os.path.join(lesson_output_dir, f"{video_name}_embedded_vector.json")
        
        print(f"Đang xử lý video {lesson_name}/{video_name}...")
        num_processed = process_video(video_dir, output_path, image_embedder)
        
        if num_processed:
            results["processed_videos"].append({
                "lesson": lesson_name,
                "video": video_name,
                "keyframes": num_processed,
                "output_file": output_path
            })
            results["status"] = "success"
            results["message"] = f"Đã xử lý video {lesson_name}/{video_name} với {num_processed} keyframes"
            results["total_videos"] = 1
        else:
            results["status"] = "warning"
            results["message"] = f"Không có keyframes nào được xử lý cho video {lesson_name}/{video_name}"
            results["total_videos"] = 0
    
    # Zip kết quả nếu có video được xử lý
    if results.get("total_videos", 0) > 0:
        zip_path = create_zip_file(output_embedded_vector_dir)
        results["zip_file"] = zip_path
    
    return results

def create_zip_file(output_embedded_vector_dir):
    """
    Tạo file zip từ thư mục embedding vectors
    
    Args:
        output_embedded_vector_dir: Thư mục chứa các file embedding vectors
        
    Returns:
        str: Đường dẫn đến file zip
    """
    import shutil
    # Lấy thư mục cơ sở và tên
    base_dir = os.path.dirname(output_embedded_vector_dir)
    dir_name = os.path.basename(output_embedded_vector_dir)
    
    # Tạo đường dẫn file zip
    zip_path = os.path.join(base_dir, f"{dir_name}.zip")
    
    # In thông tin về quá trình tạo file zip
    print(f"Đang tạo file zip của embedded vectors: {zip_path}")
    
    try:
        # Tạo file zip
        shutil.make_archive(
            os.path.join(base_dir, dir_name),  # Tên gốc của file zip
            'zip',                             # Format
            output_embedded_vector_dir          # Thư mục cần zip
        )
        print(f"Đã tạo file zip thành công tại: {zip_path}")
    except Exception as e:
        print(f"Cảnh báo: Không thể tạo file zip: {e}")
    
    return zip_path
