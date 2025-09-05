import os
import json
import glob
import sys
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel
import torch
import numpy as np


def get_caption_embedder():
    """Khởi tạo BGEM3 model để tạo embedding cho captions"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Khởi tạo BGEM3FlagModel trên thiết bị: {device}")
    return BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device=device)

def generate_caption_embeddings_batch(caption_embedder, texts, batch_size=64):
    """
    Tạo embeddings cho một batch các captions
    
    Args:
        caption_embedder: BGEM3FlagModel instance
        texts: list[str] - danh sách các captions
        batch_size: int - kích thước batch
    
    Returns:
        dict: chứa dense_vecs, lexical_weights, colbert_vecs
    """
    return caption_embedder.encode(
        texts,
        batch_size=batch_size,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )

def process_video(caption_file_path, output_embedded_vector_path, caption_embedder, batch_size=64):
    """
    Xử lý một video, tạo embeddings cho các captions từ file JSON
    
    Args:
        caption_file_path: Đường dẫn đến file JSON chứa captions của video
        output_embedded_vector_path: Đường dẫn đến file output JSON
        caption_embedder: BGEM3FlagModel instance
        batch_size: Kích thước batch để xử lý
        
    Returns:
        int: Số lượng captions đã xử lý
    """
    # Kiểm tra file caption có tồn tại không
    if not os.path.exists(caption_file_path):
        print(f"Không tìm thấy file caption: {caption_file_path}")
        return 0
    
    # Đọc dữ liệu captions từ file JSON
    try:
        with open(caption_file_path, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)
    except Exception as e:
        print(f"Lỗi khi đọc file {caption_file_path}: {e}")
        return 0
    
    if not captions_data:
        print(f"File caption trống: {caption_file_path}")
        return 0
    
    print(f"Tìm thấy {len(captions_data)} captions")
    
    # Chuẩn bị dữ liệu cho batch processing
    keyframe_names = []
    caption_texts = []
    
    for item in captions_data:
        keyframe_names.append(item["keyframe"])
        caption_texts.append(item["caption"])
    
    # Tạo embeddings bằng batch processing
    print(f"Đang tạo embeddings với batch_size={batch_size}...")
    embedding_output = generate_caption_embeddings_batch(
        caption_embedder, caption_texts, batch_size=batch_size
    )
    
    # Tạo kết quả embedded vectors
    embedded_vectors = []
    for i, keyframe_name in enumerate(keyframe_names):
        # Chuyển đổi numpy arrays thành lists để serialize JSON
        dense_vec = embedding_output["dense_vecs"][i]
        if hasattr(dense_vec, 'tolist'):
            dense_vec = dense_vec.tolist()
        
        colbert_vec = embedding_output["colbert_vecs"][i]
        if hasattr(colbert_vec, 'tolist'):
            colbert_vec = colbert_vec.tolist()
        
        lexical_weights = embedding_output["lexical_weights"][i]
        
        embedded_vectors.append({
            "keyframe": keyframe_name,
            "dense_vector": dense_vec,
            "colbert_vector": colbert_vec,
            "sparse_weights": lexical_weights
        })
    
    # Lưu kết quả vào file
    os.makedirs(os.path.dirname(output_embedded_vector_path), exist_ok=True)
    with open(output_embedded_vector_path, "w", encoding="utf-8") as f:
        json.dump(embedded_vectors, f, ensure_ascii=False, indent=2)
        
    return len(embedded_vectors)

def embed_caption(input_caption_dir, output_embedded_vector_dir, mode, lesson_name=None, video_name=None, batch_size=64):
    """
    Tạo embeddings cho captions sử dụng BGEM3FlagModel với batch processing

    Args:
        input_caption_dir: Thư mục chứa các file caption JSON
        output_embedded_vector_dir: Thư mục đầu ra để lưu các file embeddings
        mode: Chế độ xử lý ('all', 'lesson', 'video')
        lesson_name: Tên bài học khi mode là 'lesson' hoặc 'video'
        video_name: Tên video khi mode là 'video'
        batch_size: Kích thước batch cho xử lý
    
    Returns:
        Dict: Thông tin về quá trình xử lý
    """
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_embedded_vector_dir, exist_ok=True)
    print(f"Thư mục đầu ra: {output_embedded_vector_dir}")

    # Khởi tạo caption embedder
    print(f"Khởi tạo caption embedder với batch_size={batch_size}")
    caption_embedder = get_caption_embedder()
    
    # Kết quả xử lý
    results = {
        "mode": mode,
        "input_dir": input_caption_dir,
        "output_dir": output_embedded_vector_dir,
        "batch_size": batch_size,
        "processed_videos": []
    }

    # Xử lý theo mode
    if mode == "all":
        # Xử lý tất cả các bài học
        lessons = [d for d in os.listdir(input_caption_dir) 
                if os.path.isdir(os.path.join(input_caption_dir, d))]
        
        print(f"Tìm thấy {len(lessons)} bài học")
        
        total_processed = 0
        for lesson in lessons:
            lesson_dir = os.path.join(input_caption_dir, lesson)
            lesson_output_dir = os.path.join(output_embedded_vector_dir, lesson)
            os.makedirs(lesson_output_dir, exist_ok=True)
            
            # Tìm tất cả các file caption JSON trong bài học
            caption_files = glob.glob(os.path.join(lesson_dir, "*_caption.json"))
            
            if not caption_files:
                print(f"Không tìm thấy file caption nào trong bài học {lesson}")
                continue
                
            print(f"Bài học {lesson}: tìm thấy {len(caption_files)} caption files")
            
            for caption_file in caption_files:
                # Lấy tên video từ tên file caption
                base_name = os.path.basename(caption_file)
                # Ví dụ: L01_V001_caption.json -> L01_V001
                video_id = base_name.replace('_caption.json', '')
                
                output_path = os.path.join(lesson_output_dir, f"{video_id}_embedded_caption.json")
                
                print(f"Đang xử lý {lesson}/{video_id}...")
                num_processed = process_video(caption_file, output_path, caption_embedder, batch_size)
                
                if num_processed > 0:
                    total_processed += 1
                    results["processed_videos"].append({
                        "lesson": lesson,
                        "video": video_id,
                        "captions": num_processed,
                        "output_file": output_path
                    })
        
        results["status"] = "success"
        results["message"] = f"Đã xử lý {total_processed} videos từ {len(lessons)} bài học"
        results["total_videos"] = total_processed
    
    elif mode == "lesson":
        # Xử lý một bài học cụ thể
        lesson_dir = os.path.join(input_caption_dir, lesson_name)
        lesson_output_dir = os.path.join(output_embedded_vector_dir, lesson_name)
        os.makedirs(lesson_output_dir, exist_ok=True)
        
        caption_files = glob.glob(os.path.join(lesson_dir, "*_caption.json"))
        
        print(f"Bài học {lesson_name}: tìm thấy {len(caption_files)} caption files")
        
        total_processed = 0
        for caption_file in caption_files:
            # Lấy tên video từ tên file caption
            base_name = os.path.basename(caption_file)
            video_id = base_name.replace('_caption.json', '')
            
            output_path = os.path.join(lesson_output_dir, f"{video_id}_embedded_caption.json")
            
            print(f"Đang xử lý {lesson_name}/{video_id}...")
            num_processed = process_video(caption_file, output_path, caption_embedder, batch_size)
            
            if num_processed > 0:
                total_processed += 1
                results["processed_videos"].append({
                    "lesson": lesson_name,
                    "video": video_id,
                    "captions": num_processed,
                    "output_file": output_path
                })
        
        results["status"] = "success"
        results["message"] = f"Đã xử lý {total_processed} videos từ bài học {lesson_name}"
        results["total_videos"] = total_processed
    
    elif mode == "video":
        # Xử lý một video cụ thể
        caption_file = os.path.join(input_caption_dir, lesson_name, f"{lesson_name}_{video_name}_caption.json")
        
        # Tạo thư mục output cho lesson này
        lesson_output_dir = os.path.join(output_embedded_vector_dir, lesson_name)
        os.makedirs(lesson_output_dir, exist_ok=True)
        
        output_path = os.path.join(lesson_output_dir, f"{lesson_name}_{video_name}_embedded_caption.json")
        
        print(f"Đang xử lý video {lesson_name}/{video_name}...")
        num_processed = process_video(caption_file, output_path, caption_embedder, batch_size)
        
        if num_processed > 0:
            results["processed_videos"].append({
                "lesson": lesson_name,
                "video": video_name,
                "captions": num_processed,
                "output_file": output_path
            })
            results["status"] = "success"
            results["message"] = f"Đã xử lý video {lesson_name}/{video_name} với {num_processed} captions"
            results["total_videos"] = 1
        else:
            results["status"] = "warning"
            results["message"] = f"Không có captions nào được xử lý cho video {lesson_name}/{video_name}"
            results["total_videos"] = 0
    
    return results