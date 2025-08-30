import os
import sys
import json

def install_dependencies():
    import os
    
    print("Installing ASR libraries for Google Colab...")
    
    print("\n=== Installing transformers and accelerate... ===\n")
    os.system(f"{sys.executable} -m pip install transformers accelerate --quiet")
    
    print("\n=== Installing WhisperX... ===\n") 
    os.system(f"{sys.executable} -m pip install whisperx")
    
    os.system("apt-get update -qq")
    os.system("apt-get install -qq libcudnn8")
    
    try:
        import torch
        print(f"\n=== PyTorch and CUDA Information ===\n")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Error checking PyTorch info: {str(e)}")
    
    print("\nCompleted installing ASR libraries.")
    print("\nNote: If CUDA/cuDNN errors occur, try using CPU instead:")
    print("  model = whisperx.load_model('large-v2', device='cpu')")
    
    return True

def correct_transcript(corrector, transcript):
    return corrector(transcript, max_length=512, do_sample=False)[0]["generated_text"]

def transcribe_single_audio(video_path, transcriber, corrector=None):
    try:
        import whisperx
        audio = whisperx.load_audio(video_path)
        result = transcriber.transcribe(audio, batch_size=2)
        segments = result.get("segments", [])
        
        if not segments:
            return ""
            
        transcript = " ".join(seg.get("text", "") for seg in segments)
        
        if transcript.strip() and corrector is not None:
            try:
                transcript = correct_transcript(corrector, transcript)
            except Exception as e:
                print(f"Error correcting transcript: {str(e)}")
                
        return transcript
    except Exception as e:
        print(f"Error transcribing audio: {str(e)}")
        return ""

def process_video(subvideo_dir, output_transcript_dir, lesson_name, video_name, transcriber):
    video_transcripts = []
    
    if os.path.exists(subvideo_dir):
        for subvideo_file in sorted(os.listdir(subvideo_dir)):
            if not subvideo_file.endswith((".mp4", ".avi", ".mov")):
                continue
                
            subvideo_path = os.path.join(subvideo_dir, subvideo_file)
            transcript = transcribe_single_audio(subvideo_path, transcriber, None)
            
            subvideo_name = os.path.splitext(subvideo_file)[0]
            video_transcripts.append({
                "subvideo": subvideo_name,
                "transcript": transcript
            })
            
            print(f"Processed subvideo: {subvideo_name}")
    
    # Save all transcripts for this video in one JSON file
    lesson_output_dir = os.path.join(output_transcript_dir, lesson_name)
    os.makedirs(lesson_output_dir, exist_ok=True)
    
    transcript_file = os.path.join(lesson_output_dir, f"{lesson_name}_{video_name}_transcript.json")
    with open(transcript_file, 'w', encoding='utf-8') as f:
        json.dump(video_transcripts, f, indent=2, ensure_ascii=False)
    
    print(f"Saved transcript file: {transcript_file}")

def get_transcriber():
    install_dependencies()
    
    import whisperx
    import torch
    transcriber = whisperx.load_model("medium", device="cuda" if torch.cuda.is_available() else "cpu")
    return transcriber

def transcribe_audio(input_subvideo_dir, output_transcript_dir, mode, lesson_name=None):
    os.makedirs(output_transcript_dir, exist_ok=True)
    
    transcriber = get_transcriber()
    
    if mode == "lesson":
        lesson_subvideo_dir = os.path.join(input_subvideo_dir, lesson_name)
        for video_folder in sorted(os.listdir(lesson_subvideo_dir)):
            subvideo_dir = os.path.join(lesson_subvideo_dir, video_folder)
            if os.path.isdir(subvideo_dir):
                process_video(subvideo_dir, output_transcript_dir, lesson_name, video_folder, transcriber)
    else:
        for lesson_folder in sorted(os.listdir(input_subvideo_dir)):
            lesson_subvideo_dir = os.path.join(input_subvideo_dir, lesson_folder)
            if os.path.isdir(lesson_subvideo_dir):
                for video_folder in sorted(os.listdir(lesson_subvideo_dir)):
                    subvideo_dir = os.path.join(lesson_subvideo_dir, video_folder)
                    if os.path.isdir(subvideo_dir):
                        process_video(subvideo_dir, output_transcript_dir, lesson_folder, video_folder, transcriber)