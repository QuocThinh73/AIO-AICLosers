import os
import cv2
from utils import load_json


def process_video(video_path, shot_path, output_keyframe_path):
    os.makedirs(output_keyframe_path, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    shots = load_json(shot_path)["items"]
    
    for shot in shots:
        start_frame = shot["start_frame"]
        end_frame = shot["end_frame"]
        
        if end_frame < start_frame:
            continue
            
        shot_length = end_frame - start_frame + 1
        
        if shot_length <= 3:
            # If shot is too short, take all frames
            frame_indexes = list(range(start_frame, end_frame + 1))
        else:
            ratios = [0.15, 0.5, 0.85]
            frame_indexes = []
            for ratio in ratios:
                frame_idx = int(start_frame + ratio * (end_frame - start_frame))
                frame_indexes.append(frame_idx)

        # Extract and save frames
        for idx in frame_indexes:
            if idx < 0 or idx >= total_frames:
                continue
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            keyframe_name = f"{video_name}_{idx:06d}.jpg"
            cv2.imwrite(os.path.join(output_keyframe_path, keyframe_name), frame)
            
    cap.release()

def extract_keyframe(input_video_dir, input_shot_dir, output_keyframe_dir, mode, lesson_name=None):
    os.makedirs(output_keyframe_dir, exist_ok=True)
    
    if mode == "lesson":
        os.makedirs(os.path.join(output_keyframe_dir, lesson_name), exist_ok=True)
        for video_file in sorted(os.listdir(os.path.join(input_video_dir, lesson_name))):
            video_path = os.path.join(input_video_dir, lesson_name, video_file)
            shot_path = os.path.join(input_shot_dir, lesson_name, video_file.replace(".mp4", ".json"))
            output_keyframe_path = os.path.join(output_keyframe_dir, lesson_name, video_file.replace(".mp4", ""))
            process_video(video_path, shot_path, output_keyframe_path)
            
    elif mode == "all":
        for lesson_folder in sorted(os.listdir(input_video_dir)):
            os.makedirs(os.path.join(output_keyframe_dir, lesson_folder), exist_ok=True)
            for video_file in sorted(os.listdir(os.path.join(input_video_dir, lesson_folder))):
                video_path = os.path.join(input_video_dir, lesson_folder, video_file)
                shot_path = os.path.join(input_shot_dir, lesson_folder, video_file.replace(".mp4", ".json"))
                output_keyframe_path = os.path.join(output_keyframe_dir, lesson_folder, video_file.replace(".mp4", ""))
                process_video(video_path, shot_path, output_keyframe_path)
