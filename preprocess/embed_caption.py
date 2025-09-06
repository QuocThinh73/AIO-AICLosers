import os
import json
import glob
import sys
import subprocess
from tqdm import tqdm
import torch
import numpy as np
import zipfile
import shutil

def _ensure_dependencies():
    """Đảm bảo các thư viện phụ thuộc đã được cài đặt"""
    print("Kiểm tra và cài đặt các thư viện cần thiết...")
    
    # Kiểm tra xem có đang chạy trong Kaggle không
    in_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    
    required_packages = [
        "FlagEmbedding",
        "transformers",
        "accelerate",
        "sentence-transformers"
    ]
    
    for package in required_packages:
        try:
            # Thử import package để kiểm tra xem đã cài đặt chưa
            if package == "FlagEmbedding":
                try:
                    from FlagEmbedding import BGEM3FlagModel
                    print(f"Đã tìm thấy {package}")
                    continue
                except ImportError:
                    pass
            elif package == "transformers":
                try:
                    import transformers
                    print(f"Đã tìm thấy {package}")
                    continue
                except ImportError:
                    pass
            elif package == "accelerate":
                try:
                    import accelerate
                    print(f"Đã tìm thấy {package}")
                    continue
                except ImportError:
                    pass
            elif package == "sentence-transformers":
                try:
                    import sentence_transformers
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

# Import FlagEmbedding sau khi đã cài đặt
from FlagEmbedding import BGEM3FlagModel


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
            dense_vec = dense_vec.astype(float).tolist()  # Chuyển về float32/float64 trước khi tolist()
        elif hasattr(dense_vec, 'astype'):
            dense_vec = dense_vec.astype(float).tolist()
        
        colbert_vec = embedding_output["colbert_vecs"][i]
        if hasattr(colbert_vec, 'tolist'):
            colbert_vec = colbert_vec.astype(float).tolist()  # Chuyển về float32/float64 trước khi tolist()
        elif hasattr(colbert_vec, 'astype'):
            colbert_vec = colbert_vec.astype(float).tolist()
        
        # Xử lý lexical_weights - có thể là dict với numpy values
        try:
            print(f"Debug: Checking lexical_weights type for item {i}: {type(embedding_output['lexical_weights'][i])}")
            lexical_weights = embedding_output["lexical_weights"][i]
            
            # Force convert to dict with string keys for JSON serialization
            if isinstance(lexical_weights, dict):
                # Chuyển đổi tất cả values trong dict sang Python float
                sparse_vector = {}
                for k, v in lexical_weights.items():
                    # Force string key and float value
                    sparse_vector[str(k)] = float(v) if hasattr(v, 'item') else float(v) if isinstance(v, (int, float)) else v
            elif hasattr(lexical_weights, 'tolist'):
                sparse_vector = lexical_weights.astype(float).tolist()
            elif lexical_weights is None:
                sparse_vector = {}
            else:
                # Try direct conversion if possible
                sparse_vector = {"values": [float(x) for x in lexical_weights]} if hasattr(lexical_weights, '__iter__') else {}
        except Exception as e:
            print(f"Cảnh báo: Không thể xử lý lexical_weights cho {keyframe_name}: {e}")
            sparse_vector = {}
            
        item_dict = {
            "keyframe": keyframe_name,
            "dense_vector": dense_vec,
            "colbert_vector": colbert_vec,
            "sparse_vector": sparse_vector
        }
        
        # Debug - print complete item for first item
        if i == 0:
            print("Debug: First item keys:", list(item_dict.keys()))
            for k, v in item_dict.items():
                print(f"Debug: {k} type: {type(v)}")
        
        embedded_vectors.append(item_dict)
    
    # In thông tin embedding đầu tiên để quan sát
    if embedded_vectors:
        first_embedding = embedded_vectors[0]
        print(f"📋 Sample embedding output (keyframe: {first_embedding['keyframe']}):")
        print(f"  - dense_vector: shape {len(first_embedding['dense_vector'])}")
        print(f"  - colbert_vector: shape {len(first_embedding['colbert_vector'])}")
        print(f"  - sparse_vector: type {type(first_embedding['sparse_vector'])}, size {len(first_embedding['sparse_vector']) if isinstance(first_embedding['sparse_vector'], (dict, list)) else 'N/A'}")
        print(f"  - Output fields: {list(first_embedding.keys())}")
    
    # Lưu kết quả vào file
    os.makedirs(os.path.dirname(output_embedded_vector_path), exist_ok=True)
    with open(output_embedded_vector_path, "w", encoding="utf-8") as f:
        json.dump(embedded_vectors, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Đã lưu {len(embedded_vectors)} embeddings vào: {output_embedded_vector_path}")
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
    
    # Zip kết quả nếu có video được xử lý
    if results.get("total_videos", 0) > 0:
        # Zip các file đơn lẻ trước
        for processed_video in results["processed_videos"]:
            output_file = processed_video.get("output_file")
            if output_file and os.path.exists(output_file):
                zip_path = zip_single_json_file(output_file)
                if zip_path:
                    processed_video["output_file"] = zip_path
        
        # Tạo file zip tổng hợp
        zip_path = create_zip_file(output_embedded_vector_dir)
        results["zip_file"] = zip_path
    
    return results

def zip_single_json_file(json_file_path):
    """
    Tạo file zip từ một file JSON đơn lẻ
    
    Args:
        json_file_path: Đường dẫn đến file JSON cần nén
        
    Returns:
        str: Đường dẫn đến file zip
    """
    # Kiểm tra file tồn tại
    if not os.path.exists(json_file_path):
        print(f"Không tìm thấy file: {json_file_path}")
        return None
    
    # Tạo đường dẫn file zip
    zip_base_path = os.path.splitext(json_file_path)[0]
    zip_path = f"{zip_base_path}.zip"
    
    try:
        # Tạo file zip
        base_dir = os.path.dirname(json_file_path)
        file_name = os.path.basename(json_file_path)
        
        print(f"Đang nén file {file_name}...")
        # Tạo file zip chỉ chứa file JSON này
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(json_file_path, arcname=file_name)
        
        print(f"Đã tạo file zip thành công: {zip_path}")
        
        # Xóa file JSON gốc để tiết kiệm không gian
        os.remove(json_file_path)
        print(f"Đã xóa file JSON gốc: {json_file_path}")
        
    except Exception as e:
        print(f"Cảnh báo: Không thể tạo file zip cho {json_file_path}: {e}")
        return None
    
    return zip_path

def create_zip_file(output_embedded_vector_dir):
    """
    Tạo file zip từ thư mục embedding vectors (đã chứa các file zip con)
    
    Args:
        output_embedded_vector_dir: Thư mục chứa các file embedding vectors đã được nén
        
    Returns:
        str: Đường dẫn đến file zip
    """
    # Lấy thư mục cơ sở và tên
    base_dir = os.path.dirname(output_embedded_vector_dir)
    dir_name = os.path.basename(output_embedded_vector_dir)
    
    # Tạo đường dẫn file zip
    zip_path = os.path.join(base_dir, f"{dir_name}.zip")
    
    # In thông tin về quá trình tạo file zip tổng hợp
    print(f"Đang tạo file zip tổng hợp: {zip_path}")
    
    try:
        # Tìm tất cả file zip con trong thư mục
        zip_files = []
        for root, _, _ in os.walk(output_embedded_vector_dir):
            zip_files.extend(glob.glob(os.path.join(root, "*.zip")))
        
        if not zip_files:
            print(f"Không tìm thấy file zip nào trong {output_embedded_vector_dir}")
            # Sử dụng phương pháp nén truyền thống nếu không có file zip con
            shutil.make_archive(
                os.path.join(base_dir, dir_name),  # Tên gốc của file zip
                'zip',                             # Format
                output_embedded_vector_dir          # Thư mục cần zip
            )
        else:
            # Tạo file zip mới chứa tất cả các file zip con
            print(f"Tìm thấy {len(zip_files)} file zip con để nén")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for zip_file in zip_files:
                    # Lưu trữ đường dẫn tương đối trong file zip
                    rel_path = os.path.relpath(zip_file, output_embedded_vector_dir)
                    zipf.write(zip_file, arcname=rel_path)
        
        print(f"Đã tạo file zip tổng hợp thành công tại: {zip_path}")
    except Exception as e:
        print(f"Cảnh báo: Không thể tạo file zip tổng hợp: {e}")
    
    return zip_path