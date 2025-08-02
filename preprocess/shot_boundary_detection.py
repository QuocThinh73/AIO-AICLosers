import subprocess
import os
import sys
import json
import torch


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
        for video_file in sorted(os.listdir(os.path.join(input_video_dir, lesson_name))):
            if not video_file.endswith(".mp4"):
                continue
            video_path = os.path.join(input_video_dir, lesson_name, video_file)
            process_video(video_path, lesson_output_dir, predict_video)

    else:
        for lesson_folder in sorted(os.listdir(input_video_dir)):
            lesson_output_dir = os.path.join(output_shot_dir, lesson_folder)
            os.makedirs(lesson_output_dir, exist_ok=True)
            for video_file in sorted(os.listdir(os.path.join(input_video_dir, lesson_folder))):
                if not video_file.endswith(".mp4"):
                    continue
                video_path = os.path.join(input_video_dir, lesson_folder, video_file)
                process_video(video_path, lesson_output_dir, predict_video)
    
    
    
        