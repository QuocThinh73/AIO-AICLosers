import subprocess
import os
import sys
import json
import torch


def get_predict_video():
    subprocess.run(["git", "clone", "https://github.com/SlimRG/transnetv2pt.git"], check=True)
    sys.path.insert(0, os.path.abspath("transnetv2pt"))
    from transnetv2pt import predict_video
    return predict_video

def process_video(video_path, output_shot_path, predict_video):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scenes = predict_video(video_path, device=device, show_progressbar=True)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file = os.path.join(output_shot_path, f"{video_name}_shots.json")
    
    items = []
    for start_frame, end_frame in scenes:
        items.append({
            "start_frame": start_frame,
            "end_frame": end_frame,
        })
    
    data = {
        "total": len(items),
        "items": items
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def detect_shot_boundary(input_video_dir, output_shot_dir, mode, lesson_name=None):
    os.makedirs(output_shot_dir, exist_ok=True)
    
    predict_video = get_predict_video()
    
    if mode == "lesson":
        os.makedirs(os.path.join(output_shot_dir, lesson_name), exist_ok=True)
        for video_file in sorted(os.listdir(os.path.join(input_video_dir, lesson_name))):
            video_path = os.path.join(input_video_dir, lesson_name, video_file)
            output_shot_path = os.path.join(output_shot_dir, lesson_name, video_file.replace(".mp4", ".json"))
            process_video(video_path, output_shot_path, predict_video)

    else:
        for lesson_folder in sorted(os.listdir(input_video_dir)):
            os.makedirs(os.path.join(output_shot_dir, lesson_folder), exist_ok=True)
            for video_file in sorted(os.listdir(os.path.join(input_video_dir, lesson_folder))):
                video_path = os.path.join(input_video_dir, lesson_folder, video_file)
                output_shot_path = os.path.join(output_shot_dir, lesson_folder, video_file.replace(".mp4", ".json"))
                process_video(video_path, output_shot_path, predict_video)
    
    
    
        