
import os
import sys
import json
from tqdm import tqdm

def install_dependencies():
    """Cài đặt các thư viện và phụ thuộc cần thiết cho ASR"""
    import subprocess
    import platform
    import os
    
    # Kiểm tra nếu đang chạy trên Google Colab
    is_colab = 'google.colab' in sys.modules
    if is_colab:
        print("Phát hiện môi trường Google Colab. Cài đặt các phụ thuộc tương thích.")
    
    # Danh sách các gói cần thiết cho WhisperX
    dependencies = [
        "torch",  # Cài đặt PyTorch trước vì các gói khác phụ thuộc vào nó
        "torchaudio",
        "transformers",
        "accelerate",
        "ffmpeg-python",
        "pyannote.audio",
        "whisperx",
    ]
    
    print("Kiểm tra và cài đặt thư viện...")
    
    # Kiểm tra CUDA availability trước
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            print(f"  [✓] CUDA đã có sẵn (phiên bản {cuda_version})")
        else:
            print("  [✗] CUDA không khả dụng - WhisperX sẽ chạy chậ hơn trên CPU")
    except:
        print("  [✗] Chưa cài đặt PyTorch")
    
    # Cấu hình CUDA cho tensorflow nếu đang dùng Colab
    if is_colab:
        print("  Cấu hình môi trường CUDA trên Colab...")
        try:
            # Enable TF32 for PyTorch to address the warning
            import torch
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("  [✓] Đã kích hoạt TensorFloat-32 (TF32) cho PyTorch")
        except:
            pass
    
    # Cài đặt các gói phụ thuộc
    for dep in dependencies:
        try:
            # Thử import gói
            if dep == "ffmpeg-python":
                import ffmpeg
            else:
                __import__(dep.split('-')[0])
            print(f"  [✓] {dep} đã được cài đặt")
        except ImportError:
            print(f"  [✗] Đang cài đặt {dep}...")
            try:
                # Nếu là PyTorch, sử dụng lệnh cài đặt riêng cho Colab
                if dep == "torch" and is_colab:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "torch==2.0.1+cu118", "torchvision==0.15.2+cu118", "-f", "https://download.pytorch.org/whl/torch_stable.html"],
                        stdout=subprocess.DEVNULL
                    )
                else:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", dep],
                        stdout=subprocess.DEVNULL
                    )
                print(f"      Đã cài đặt thành công {dep}")
            except Exception as e:
                print(f"      Lỗi khi cài đặt {dep}: {str(e)}")
    
    # Kiểm tra libcudnn_ops_infer.so.8 nếu đang chạy trên Colab
    if is_colab:
        try:
            # Cài đặt cudnn nếu cần
            print("  Kiểm tra cuDNN...")
            missing_cudnn = False
            
            # Kiểm tra xem có libcudnn_ops_infer.so.8 chưa
            if not os.path.exists('/usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so.8'):
                missing_cudnn = True
            
            if missing_cudnn:
                print("  [✗] Thiếu thư viện cuDNN. Đang cài đặt...")  
                subprocess.check_call(
                    ["apt-get", "update", "-qq"], 
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                subprocess.check_call(
                    ["apt-get", "install", "-qq", "libcudnn8"], 
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print("      Đã cài đặt libcudnn8")
            else:
                print("  [✓] cuDNN đã được cài đặt")  
        except Exception as e:
            print(f"      Lỗi khi kiểm tra/cài đặt cuDNN: {str(e)}")
            print("      Bạn có thể cần chạy: !apt-get update && apt-get install -y libcudnn8")
    
    print("Hoàn tất kiểm tra và cài đặt thư viện.")
    return True

def correct_transcript(corrector, transcript):
    """
    Sửa lỗi trong bản ghi âm sử dụng mô hình hiệu chỉnh tiếng Việt
    
    Args:
        corrector: Mô hình hiệu chỉnh văn bản
        transcript: Văn bản cần hiệu chỉnh
    
    Returns:
        str: Văn bản đã được hiệu chỉnh
    """
    return corrector(transcript, max_length=512, do_sample=False)[0]["generated_text"]

def transcribe_audio(video_path, transcriber, corrector=None):
    """
    Thực hiện chuyển đổi âm thanh trong video thành văn bản và hiệu chỉnh nếu có
    
    Args:
        video_path: Đường dẫn tới tệp video
        transcriber: Mô hình WhisperX để chuyển âm thanh thành văn bản
        corrector: Mô hình hiệu chỉnh văn bản (mặc định: None)
        
    Returns:
        str: Văn bản đã được chuyển đổi từ âm thanh
    """
    try:
        import whisperx
        audio = whisperx.load_audio(video_path)
        result = transcriber.transcribe(audio, batch_size=2)
        segments = result.get("segments", [])
        
        if not segments:
            return ""
            
        # Nối các đoạn văn bản lại với nhau
        transcript = " ".join(seg.get("text", "") for seg in segments)
        
        # Hiệu chỉnh văn bản nếu có corrector
        if transcript.strip() and corrector is not None:
            try:
                transcript = correct_transcript(corrector, transcript)
            except Exception as e:
                print(f"Lỗi khi hiệu chỉnh văn bản: {str(e)}")
                
        return transcript
    except Exception as e:
        print(f"Lỗi khi chuyển đổi âm thanh: {str(e)}")
        return ""

# Removed unused audio loading functions as we now use whisperx.load_audio directly

def process_lesson_asr(lesson_path, transcript_folder, lesson_name=None):
    try:
        if not os.path.exists(lesson_path):
            return {"status": "error", "message": f"Không tìm thấy thư mục bài học: {lesson_path}"}
        
        install_dependencies()
        import torch
        
        # Tạo thư mục đầu ra
        lesson_name = lesson_name or os.path.basename(lesson_path)
        transcript_lesson_path = os.path.join(transcript_folder, lesson_name)
        os.makedirs(transcript_lesson_path, exist_ok=True)
        
        # Khởi tạo mô hình
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Đang khởi tạo mô hình trên thiết bị {device}...")
        
        # Khởi tạo WhisperX
        try:
            import whisperx
            transcriber = whisperx.load_model("large-v2", device=device, compute_type="int8")
            print(f"Đã khởi tạo thành công mô hình WhisperX trên {device}")
        except Exception as e:
            return {"status": "error", "message": f"Không thể khởi tạo mô hình WhisperX: {str(e)}"}
        
        # Khởi tạo bộ hiệu chỉnh tiếng Việt
        print("Đang khởi tạo mô hình hiệu chỉnh tiếng Việt...")
        try:
            from transformers import pipeline
            corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction")
            print(f"Đã khởi tạo thành công mô hình hiệu chỉnh tiếng Việt")
        except Exception as e:
            print(f"Không thể khởi tạo mô hình hiệu chỉnh tiếng Việt: {str(e)}")
            corrector = None
        
        results = {}
        total_videos_processed = 0
        total_subvideos_processed = 0
        
        for video_folder in sorted(os.listdir(lesson_path)):
            video_folder_path = os.path.join(lesson_path, video_folder)
            
            if not os.path.isdir(video_folder_path):
                continue
                
            print(f"Đang xử lý {video_folder}")
            video_transcripts = []
            
            for subvideo in tqdm(sorted(os.listdir(video_folder_path))):
                if not subvideo.endswith(".mp4"):
                    continue
                    
                subvideo_path = os.path.join(video_folder_path, subvideo)
                
                # Sử dụng hàm transcribe_audio để chuyển đổi âm thanh thành văn bản
                transcript = transcribe_audio(subvideo_path, transcriber, corrector)
                
                video_transcripts.append({
                    "subvideo": subvideo_path,
                    "transcript": transcript,
                })
                
                total_subvideos_processed += 1
                
            if video_transcripts:
                output_json_path = os.path.join(transcript_lesson_path, f"{lesson_name}_{video_folder}_transcript.json")
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(video_transcripts, f, ensure_ascii=False, indent=4)
                
                results[video_folder] = {
                    "path": output_json_path,
                    "subvideos_processed": len(video_transcripts)
                }
                
                total_videos_processed += 1
        
        return {
            "status": "success", 
            "message": f"Đã xử lý thành công {total_videos_processed} thư mục video với tổng số {total_subvideos_processed} subvideo",
            "results": results
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Lỗi khi xử lý ASR: {str(e)}"}

def process_all_lessons_asr(input_video_dir, transcript_folder):
    """
    Xử lý ASR cho tất cả các bài học trong thư mục
    
    Args:
        input_video_dir: Thư mục chứa tất cả các bài học
        transcript_folder: Thư mục đầu ra cho bản ghi âm
        
    Returns:
        dict: Kết quả của quá trình xử lý
    """
    try:
        # Kiểm tra đường dẫn đầu vào
        if not os.path.exists(input_video_dir):
            print(f"Lỗi: Không tìm thấy thư mục đầu vào: {input_video_dir}")
            return {"status": "error", "message": f"Không tìm thấy thư mục đầu vào: {input_video_dir}"}
        
        # Tạo thư mục đầu ra nếu chưa có
        os.makedirs(transcript_folder, exist_ok=True)
        
        results = {}
        total_lessons_processed = 0
        total_videos_processed = 0
        total_subvideos_processed = 0
        
        # Xử lý từng bài học
        lessons = sorted([d for d in os.listdir(input_video_dir) 
                        if os.path.isdir(os.path.join(input_video_dir, d))])
        
        print(f"Tìm thấy {len(lessons)} bài học để xử lý")
        
        for lesson_name in tqdm(lessons, desc="Xử lý bài học"):
            lesson_path = os.path.join(input_video_dir, lesson_name)
            print(f"\n=== Đang xử lý bài học {lesson_name} ===")
            lesson_result = process_lesson_asr(lesson_path, transcript_folder, lesson_name)
                
            if lesson_result["status"] == "success":
                results[lesson_name] = lesson_result["results"]
                total_lessons_processed += 1
                lesson_videos = len(lesson_result["results"])
                lesson_subvideos = sum(item["subvideos_processed"] for item in lesson_result["results"].values())
                total_videos_processed += lesson_videos
                total_subvideos_processed += lesson_subvideos
                print(f"Hoàn thành bài học {lesson_name}: {lesson_videos} video, {lesson_subvideos} subvideo")
            else:
                print(f"Lỗi khi xử lý bài học {lesson_name}: {lesson_result['message']}")
        
        return {
            "status": "success", 
            "message": f"Đã xử lý thành công {total_lessons_processed} bài học với {total_videos_processed} video và {total_subvideos_processed} subvideo",
            "results": results
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Lỗi khi xử lý ASR: {str(e)}"}