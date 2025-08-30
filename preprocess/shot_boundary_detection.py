import subprocess
import os
import sys
import json
import torch
import numpy as np
import cv2
from tqdm import tqdm


def check_video_codec(video_path):
    """Check if video uses AV1 codec using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_streams', '-select_streams', 'v:0', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        if 'streams' in data and len(data['streams']) > 0:
            codec_name = data['streams'][0].get('codec_name', '').lower()
            return codec_name == 'av1'
        return False
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        return False


def extract_frames_with_ffmpeg(video_path, target_height=27, target_width=48, show_progressbar=False):
    """Extract frames using FFmpeg with software decoding for AV1 videos"""
    import tempfile
    import shutil
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Get total frame count first
        cmd_count = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', '-select_streams', 'v:0', video_path
        ]
        result = subprocess.run(cmd_count, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        total_frames = 0
        if 'streams' in data and len(data['streams']) > 0:
            stream = data['streams'][0]
            if 'nb_frames' in stream:
                total_frames = int(stream['nb_frames'])
            else:
                # Fallback: estimate from duration and fps
                duration = float(stream.get('duration', 0))
                fps = eval(stream.get('avg_frame_rate', '25/1'))
                total_frames = int(duration * fps)
        
        # Extract frames using FFmpeg with software decoding
        output_pattern = os.path.join(temp_dir, 'frame_%06d.png')
        cmd_extract = [
            'ffmpeg', '-v', 'quiet', '-hwaccel', 'none',  # Force software decoding
            '-i', video_path,
            '-vf', f'scale={target_width}:{target_height}',
            '-y', output_pattern
        ]
        
        print(f"Extracting frames with FFmpeg (software decoding)...")
        subprocess.run(cmd_extract, check=True)
        
        # Read extracted frames
        frames = []
        frame_files = sorted([f for f in os.listdir(temp_dir) if f.startswith('frame_')])
        
        progress_bar = tqdm(total=len(frame_files), desc="Loading frames", unit="frame") if show_progressbar else None
        
        for frame_file in frame_files:
            frame_path = os.path.join(temp_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is not None:
                frames.append(frame)
            
            if progress_bar:
                progress_bar.update(1)
        
        if progress_bar:
            progress_bar.close()
            
        print(f"Extracted {len(frames)} frames using FFmpeg")
        return np.array(frames)
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def patch_transnetv2_inference():
    """Patch TransNet inference.py to handle AV1 videos with FFmpeg"""
    inference_path = "transnetv2pt/transnetv2pt/inference.py"
    
    if not os.path.exists(inference_path):
        print(f"Warning: {inference_path} not found, patch will be applied when TransNet is loaded")
        return
    
    # Read current inference.py
    with open(inference_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already patched
    if 'check_video_codec' in content:
        print("TransNet inference.py already patched for AV1 support")
        return
    
    # Create patched version
    patch_imports = '''import subprocess
import json
import tempfile
import shutil
'''
    
    patch_functions = '''
def check_video_codec(video_path):
    """Check if video uses AV1 codec using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_streams', '-select_streams', 'v:0', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        if 'streams' in data and len(data['streams']) > 0:
            codec_name = data['streams'][0].get('codec_name', '').lower()
            return codec_name == 'av1'
        return False
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        return False


def extract_frames_with_ffmpeg_av1(video_path, target_height=27, target_width=48, show_progressbar=False):
    """Extract frames using FFmpeg with software decoding for AV1 videos"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Extract frames using FFmpeg with software decoding
        output_pattern = os.path.join(temp_dir, 'frame_%06d.png')
        cmd_extract = [
            'ffmpeg', '-v', 'quiet', '-hwaccel', 'none',  # Force software decoding
            '-i', video_path,
            '-vf', f'scale={target_width}:{target_height}',
            '-y', output_pattern
        ]
        
        logger.info(f"Extracting frames with FFmpeg (software decoding) from: {video_path}")
        subprocess.run(cmd_extract, check=True)
        
        # Read extracted frames
        frames = []
        frame_files = sorted([f for f in os.listdir(temp_dir) if f.startswith('frame_')])
        
        progress_bar = tqdm(total=len(frame_files), desc="Loading frames", unit="frame") if show_progressbar else None
        
        for frame_file in frame_files:
            frame_path = os.path.join(temp_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is not None:
                frames.append(frame)
            
            if progress_bar:
                progress_bar.update(1)
        
        if progress_bar:
            progress_bar.close()
            
        logger.info(f"Extracted {len(frames)} frames using FFmpeg")
        return np.array(frames)
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

'''
    
    # Find the original extract_frames_with_opencv function and modify it
    original_function = '''def extract_frames_with_opencv(video_path: str, target_height: int = 27, target_width: int = 48, show_progressbar: bool = False):'''
    
    patched_function = '''def extract_frames_with_opencv(video_path: str, target_height: int = 27, target_width: int = 48, show_progressbar: bool = False):
    """
    Extracts frames from a video using OpenCV with optional CUDA support and progress tracking.
    For AV1 videos, falls back to FFmpeg with software decoding.
    """
    # Check if video uses AV1 codec
    if check_video_codec(video_path):
        logger.info(f"AV1 video detected, using FFmpeg for extraction: {video_path}")
        return extract_frames_with_ffmpeg_av1(video_path, target_height, target_width, show_progressbar)
    
    # Original OpenCV implementation for non-AV1 videos'''
    
    # Add imports at the top
    if 'import subprocess' not in content:
        content = content.replace('import logging', f'import logging\n{patch_imports}')
    
    # Add helper functions before the original extract_frames_with_opencv
    func_pos = content.find('def extract_frames_with_opencv')
    if func_pos != -1:
        content = content[:func_pos] + patch_functions + '\n' + content[func_pos:]
    
    # Replace the function signature and add AV1 check
    content = content.replace(original_function, patched_function)
    
    # Write patched file
    with open(inference_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("TransNet inference.py patched successfully for AV1 support")


def get_predict_video():
    repo_path = "transnetv2pt"
    
    # Kiểm tra nếu thư mục đã tồn tại
    if not os.path.exists(repo_path):
        try:
            print(f"Cloning repository {repo_path}...")
            subprocess.run(["git", "clone", f"https://github.com/SlimRG/{repo_path}.git"], check=True)
            print(f"Repository {repo_path} cloned successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e}")
            if os.path.exists(repo_path):
                print(f"But directory {repo_path} exists, attempting to use it anyway.")
            else:
                raise
    else:
        print(f"Repository {repo_path} already exists, using local copy.")
    
    # Apply AV1 patch to TransNet inference.py
    patch_transnetv2_inference()
        
    sys.path.insert(0, os.path.abspath(repo_path))
    
    try:
        from transnetv2pt import predict_video
        return predict_video
    except ImportError as e:
        print(f"Error importing predict_video: {e}")
        raise

def process_video(video_path, output_shot_path, predict_video):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scenes = predict_video(video_path, device=device, show_progressbar=True)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file = os.path.join(output_shot_path, f"{video_name}_shots.json")
    
    
    items = []
    for start_frame, end_frame in scenes:
        
        items.append({
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
        })
    
    data = {
        "total": len(items),
        "items": items
    }
    
    print(f"Lưu kết quả shots vào {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def detect_shot_boundary(input_video_dir, output_shot_dir, mode, lesson_name=None):
    os.makedirs(output_shot_dir, exist_ok=True)
    
    predict_video = get_predict_video()
    
    if mode == "lesson":
        lesson_output_dir = os.path.join(output_shot_dir, lesson_name)
        os.makedirs(lesson_output_dir, exist_ok=True)
        lesson_path = os.path.join(input_video_dir, lesson_name)
        
        for video_folder in sorted(os.listdir(lesson_path)):
            video_folder_path = os.path.join(lesson_path, video_folder)
            if os.path.isdir(video_folder_path):
                video_folder_output = os.path.join(lesson_output_dir, video_folder)
                os.makedirs(video_folder_output, exist_ok=True)
                
                for video_file in sorted(os.listdir(video_folder_path)):
                    if not video_file.endswith(".mp4"):
                        continue
                    video_path = os.path.join(video_folder_path, video_file)
                    process_video(video_path, video_folder_output, predict_video)
            elif video_folder.endswith(".mp4"):
                video_path = video_folder_path
                process_video(video_path, lesson_output_dir, predict_video)

    else:  # mode == "all"
        for lesson_folder in sorted(os.listdir(input_video_dir)):
            lesson_path = os.path.join(input_video_dir, lesson_folder)
            if not os.path.isdir(lesson_path):
                continue
                
            lesson_output_dir = os.path.join(output_shot_dir, lesson_folder)
            os.makedirs(lesson_output_dir, exist_ok=True)
            
            for video_folder in sorted(os.listdir(lesson_path)):
                video_folder_path = os.path.join(lesson_path, video_folder)
                if os.path.isdir(video_folder_path):
                    video_folder_output = os.path.join(lesson_output_dir, video_folder)
                    os.makedirs(video_folder_output, exist_ok=True)
                    
                    for video_file in sorted(os.listdir(video_folder_path)):
                        if not video_file.endswith(".mp4"):
                            continue
                        video_path = os.path.join(video_folder_path, video_file)
                        process_video(video_path, video_folder_output, predict_video)
                elif video_folder.endswith(".mp4"):
                    video_path = video_folder_path
                    process_video(video_path, lesson_output_dir, predict_video)
    
    
    
        