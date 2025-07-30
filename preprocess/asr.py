
import os
import sys
import json
import glob
from tqdm import tqdm

def ensure_whisperx_dependencies():
    """
    Đảm bảo các thư viện cần thiết cho ASR đã được cài đặt
    với các phiên bản tương thích
    """
    try:
        # Kiểm tra phiên bản NumPy
        import numpy
        numpy_version = numpy.__version__
        print(f"NumPy version detected: {numpy_version}")
        
        # Nếu là NumPy 2.x, hạ cấp xuống 1.x để tương thích
        if numpy_version.startswith('2'):
            print("Downgrading NumPy from 2.x to 1.26.4 for compatibility...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.26.4", "--force-reinstall"])
            print("NumPy downgraded successfully")
            # Tải lại module NumPy
            import importlib
            importlib.reload(numpy)
            print(f"New NumPy version: {numpy.__version__}")
    except Exception as e:
        print(f"Warning: Could not check/fix NumPy version: {str(e)}")
    
    # Cài đặt các thư viện theo thứ tự phụ thuộc
    dependencies = [
        "numpy==1.23.5",  # Phiên bản cũ hơn, tương thích tốt với nhiều thư viện ML
        "scipy==1.10.1",  # Phiên bản tương thích với PyTorch và Pyannote
        "transformers==4.36.2",  # Phiên bản đã biết là hoạt động
        "accelerate",
        "pyannote.audio==3.1.1",  # Cố định phiên bản pyannote
        "pytorch_lightning==2.1.0",  # Cố định phiên bản pytorch_lightning
        "whisperx==3.1.1",  # Cố định phiên bản whisperx để đảm bảo tương thích
    ]
    
    for dep in dependencies:
        try:
            if "numpy" in dep:
                import numpy
                print(f"NumPy version: {numpy.__version__}")
            elif "transformers" in dep:
                from transformers import __version__ as transformers_version
                print(f"Transformers version: {transformers_version}")
                # Kiểm tra phiên bản transformers
                from transformers import pipeline
            elif "accelerate" in dep:
                import accelerate
            elif "whisperx" in dep:
                import whisperx
        except ImportError:
            print(f"Đang cài đặt {dep}...")
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "--ignore-installed"])
                print(f"Đã cài đặt {dep}")
            except Exception as install_error:
                print(f"Lỗi khi cài đặt {dep}: {str(install_error)}")
                print("Thử phương pháp cài đặt thay thế...")
                try:
                    # Thử cài đặt không chỉ định phiên bản nếu cài đặt cụ thể thất bại
                    dep_name = dep.split('==')[0] if '==' in dep else dep
                    subprocess.check_call([sys.executable, "-m", "pip", "install", dep_name])
                    print(f"Đã cài đặt {dep_name} (không chỉ định phiên bản)")
                except Exception as alt_error:
                    print(f"Không thể cài đặt {dep_name}: {str(alt_error)}")
                    # Tiếp tục vì có thể thư viện đã được cài đặt trước đó

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

def transcribe_audio(video_path, transcriber, corrector):
    """
    Thực hiện chuyển đổi âm thanh trong video thành văn bản và hiệu chỉnh
    
    Args:
        video_path: Đường dẫn tới tệp video
        transcriber: Mô hình WhisperX để chuyển âm thanh thành văn bản
        corrector: Mô hình hiệu chỉnh văn bản
        
    Returns:
        str: Văn bản đã được chuyển đổi từ âm thanh và hiệu chỉnh
    """
    try:
        import whisperx
        audio = whisperx.load_audio(video_path)
        transcripts = transcriber.transcribe(audio, batch_size=2)
        transcripts = [correct_transcript(corrector, transcript['text']) for transcript in transcripts['segments']]

        return " ".join(transcripts)
    except Exception as e:
        print(f"Lỗi khi chuyển đổi âm thanh thành văn bản cho {video_path}: {str(e)}")
        return ""

class KaggleFallbackTranscriber:
    """
    Lớp fallback cho transcriber khi chạy trên Kaggle gặp lỗi
    Sử dụng phương pháp transcribe đơn giản hơn
    """
    def __init__(self, model="base", device="cpu"):
        self.model_name = model
        self.device = device
        self.model = None
        try:
            import torch
            import transformers
            from transformers import pipeline
            print(f"Khởi tạo Whisper fallback model ({model}) trên {device}...")
            self.model = pipeline(
                "automatic-speech-recognition",
                model=f"openai/whisper-{model}",
                device=device,
                chunk_length_s=30,
            )
            print("Khởi tạo thành công mô hình Whisper fallback")
        except Exception as e:
            print(f"Không thể khởi tạo mô hình Whisper fallback: {str(e)}")
    
    def transcribe(self, audio, batch_size=2):
        try:
            if self.model is None:
                return {"segments": [{"text": ""}]}
                
            result = self.model(audio)
            text = result["text"] if isinstance(result, dict) else result
            
            # Convert to WhisperX-like format
            return {"segments": [{"text": text}]}
        except Exception as e:
            print(f"Lỗi khi chuyển đổi âm thanh: {str(e)}")
            return {"segments": [{"text": ""}]}

def load_audio_fallback(audio_path):
    """
    Tải âm thanh sử dụng các phương pháp dự phòng khác nhau
    """
    try:
        try:
            # Thử cách 1: Sử dụng whisperx
            import whisperx
            return whisperx.load_audio(audio_path)
        except ImportError:
            # Thử cách 2: Sử dụng librosa
            try:
                import librosa
                return librosa.load(audio_path, sr=16000)[0]
            except ImportError:
                # Thử cách 3: Sử dụng soundfile
                import soundfile as sf
                audio, sr = sf.read(audio_path)
                if sr != 16000:
                    # Resample to 16kHz (Whisper's expected sample rate)
                    try:
                        import resampy
                        audio = resampy.resample(audio, sr, 16000)
                    except ImportError:
                        # Simple resample
                        import numpy as np
                        audio = np.array(audio[::int(sr/16000)])
                return audio
    except Exception as e:
        print(f"Không thể đọc tệp âm thanh {audio_path}: {str(e)}")
        import numpy as np
        return np.zeros(1600)  # Return empty audio

def process_lesson_asr(lesson_path, transcript_folder, lesson_name=None):
    """
    Xử lý ASR cho một bài học cụ thể
    
    Args:
        lesson_path: Đường dẫn đến thư mục bài học
        transcript_folder: Thư mục đầu ra cho bản ghi âm
        lesson_name: Tên bài học (tùy chọn, mặc định lấy từ tên thư mục)
        
    Returns:
        dict: Kết quả của quá trình xử lý
    """
    try:
        if not os.path.exists(lesson_path):
            return {"status": "error", "message": f"Không tìm thấy thư mục bài học: {lesson_path}"}
        
        ensure_whisperx_dependencies()
        import torch
        
        # Tạo thư mục đầu ra
        lesson_name = lesson_name or os.path.basename(lesson_path)
        transcript_lesson_path = os.path.join(transcript_folder, lesson_name)
        os.makedirs(transcript_lesson_path, exist_ok=True)
        
        # Khởi tạo mô hình
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Đang khởi tạo mô hình trên thiết bị {device}...")
        
        # Thử khởi tạo WhisperX
        transcriber = None
        try:
            import whisperx
            transcriber = whisperx.load_model("large-v2", device=device, compute_type="int8")
            print("Đã khởi tạo thành công mô hình WhisperX")
        except Exception as e:
            print(f"Không thể khởi tạo mô hình WhisperX: {str(e)}")
            print("Chuyển sang sử dụng mô hình dự phòng...")
            transcriber = KaggleFallbackTranscriber(model="large", device=device)
        
        # Khởi tạo bộ hiệu chỉnh tiếng Việt
        print("Đang khởi tạo mô hình hiệu chỉnh tiếng Việt...")
        corrector = None
        try:
            from transformers import pipeline
            corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction")
            print("Đã khởi tạo thành công mô hình hiệu chỉnh tiếng Việt")
        except Exception as e:
            print(f"Không thể khởi tạo mô hình hiệu chỉnh tiếng Việt: {str(e)}")
            print("Tiếp tục mà không sử dụng hiệu chỉnh")
        
        # Hàm hiệu chỉnh an toàn
        def safe_correct(text):
            if corrector and text.strip():
                try:
                    return correct_transcript(corrector, text)
                except Exception as e:
                    print(f"Lỗi khi hiệu chỉnh văn bản: {str(e)}")
            return text
        
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
                
                # Sử dụng cả hai phương pháp tải âm thanh
                audio = None
                try:
                    # Thử sử dụng WhisperX nếu đã khởi tạo thành công
                    if isinstance(transcriber, KaggleFallbackTranscriber):
                        audio = load_audio_fallback(subvideo_path)
                    else:
                        import whisperx
                        audio = whisperx.load_audio(subvideo_path)
                except Exception as e:
                    print(f"Lỗi khi tải âm thanh bằng WhisperX: {str(e)}")
                    print("Chuyển sang phương pháp dự phòng...")
                    audio = load_audio_fallback(subvideo_path)
                
                # Thực hiện chuyển đổi văn bản
                transcript = ""
                try:
                    # Transcribe audio
                    result = transcriber.transcribe(audio, batch_size=2)
                    segments = result.get("segments", [])
                    
                    if segments:
                        # Nối các đoạn văn bản lại với nhau
                        segment_texts = [seg.get("text", "") for seg in segments]
                        transcript = " ".join(segment_texts)
                        
                        # Hiệu chỉnh văn bản nếu có corrector
                        if transcript.strip() and corrector is not None:
                            transcript = safe_correct(transcript)
                except Exception as e:
                    print(f"Lỗi khi chuyển đổi âm thanh: {str(e)}")
                    transcript = ""
                
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
        if not os.path.exists(input_video_dir):
            return {"status": "error", "message": f"Không tìm thấy thư mục đầu vào: {input_video_dir}"}
        
        os.makedirs(transcript_folder, exist_ok=True)
        
        results = {}
        total_lessons_processed = 0
        total_videos_processed = 0
        total_subvideos_processed = 0
        
        for lesson_name in sorted(os.listdir(input_video_dir)):
            lesson_path = os.path.join(input_video_dir, lesson_name)
            
            if not os.path.isdir(lesson_path):
                continue
                
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