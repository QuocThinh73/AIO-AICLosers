import os
import sys
import json
from tqdm import tqdm

def install_dependencies():
    """Cài đặt các thư viện và phụ thuộc cần thiết cho ASR"""
    import os
    
    print("Bắt đầu cài đặt thư viện cho ASR trên Google Colab...")
    
    # Sử dụng cách cài đặt đơn giản hơn, trực tiếp tương tự như notebook gốc
    print("\n=== Cài đặt transformers và accelerate... ===\n")
    os.system(f"{sys.executable} -m pip install transformers accelerate --quiet")
    
    print("\n=== Cài đặt WhisperX... ===\n") 
    os.system(f"{sys.executable} -m pip install whisperx")
    
    os.system("apt-get update -qq")
    os.system("apt-get install -qq libcudnn8")
    
    try:
        import torch
        print(f"\n=== Thông tin PyTorch và CUDA ===\n")
        print(f"PyTorch phiên bản: {torch.__version__}")
        print(f"CUDA khả dụng: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA phiên bản: {torch.version.cuda}")
            print(f"Thiết bị CUDA: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Lỗi khi kiểm tra thông tin PyTorch: {str(e)}")
    
    print("\nHoàn tất cài đặt các thư viện cho ASR.")
    print("\nLưu ý: Nếu gặp lỗi về CUDA/cuDNN, hãy thử sử dụng CPU thay thế:")
    print("  model = whisperx.load_model('large-v2', device='cpu')")
    
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

def transcribe_single_audio(video_path, transcriber, corrector=None):
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
            print(f"Lỗi khi khởi tạo mô hình WhisperX với GPU: {str(e)}")
            print("Thử lại với CPU...")
            try:
                device = "cpu"
                transcriber = whisperx.load_model("large-v2", device=device, compute_type="float32")
                print(f"Đã khởi tạo thành công mô hình WhisperX trên CPU")
            except Exception as e2:
                return {"status": "error", "message": f"Không thể khởi tạo mô hình WhisperX trên CPU: {str(e2)}"}
         
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
                transcript = transcribe_single_audio(subvideo_path, transcriber, corrector)
                
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

def process_video(video_path, output_transcript_dir, lesson_name):
    install_dependencies()
    
    import whisperx
    import torch
    transcriber = whisperx.load_model("medium", device="cuda" if torch.cuda.is_available() else "cpu")
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Process subvideos in this video directory
    video_folder_path = os.path.dirname(video_path).replace("videos", "subvideos")
    if os.path.exists(video_folder_path):
        for subvideo in sorted(os.listdir(video_folder_path)):
            if not subvideo.endswith((".mp4", ".avi", ".mov")):
                continue
                
            subvideo_path = os.path.join(video_folder_path, subvideo)
            transcript = transcribe_single_audio(subvideo_path, transcriber, None)
            
            # Save transcript
            lesson_output_dir = os.path.join(output_transcript_dir, lesson_name)
            os.makedirs(lesson_output_dir, exist_ok=True)
            
            transcript_file = os.path.join(lesson_output_dir, f"{os.path.splitext(subvideo)[0]}_transcript.txt")
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(transcript)

def transcribe_audio(input_video_dir, output_transcript_dir, mode, lesson_name=None):
    os.makedirs(output_transcript_dir, exist_ok=True)
    
    if mode == "lesson":
        lesson_video_dir = os.path.join(input_video_dir, lesson_name)
        for video_file in sorted(os.listdir(lesson_video_dir)):
            if not video_file.endswith(".mp4"):
                continue
            video_path = os.path.join(lesson_video_dir, video_file)
            process_video(video_path, output_transcript_dir, lesson_name)
    else:
        for lesson_folder in sorted(os.listdir(input_video_dir)):
            lesson_video_dir = os.path.join(input_video_dir, lesson_folder)
            if os.path.isdir(lesson_video_dir):
                for video_file in sorted(os.listdir(lesson_video_dir)):
                    if not video_file.endswith(".mp4"):
                        continue
                    video_path = os.path.join(lesson_video_dir, video_file)
                    process_video(video_path, output_transcript_dir, lesson_folder)