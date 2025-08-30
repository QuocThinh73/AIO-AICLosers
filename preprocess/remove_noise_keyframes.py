import os
import json
import glob
from .utils import load_json, parse_lesson_video_name, get_lesson_directories


def load_news_anchor_predictions(news_anchor_dir):
    predictions = {}
    
    # Get all lesson directories
    lesson_dirs = get_lesson_directories(news_anchor_dir)
    
    for lesson_num, lesson_name, lesson_path in lesson_dirs:
        # Find all news anchor JSON files in this lesson
        pattern = os.path.join(lesson_path, f"{lesson_name}_*_news_anchor.json")
        json_files = glob.glob(pattern)
        
        for json_file in json_files:
            # Extract video name from filename
            filename = os.path.basename(json_file)
            video_name = filename.replace("_news_anchor.json", "")
            
            # Load predictions
            data = load_json(json_file)
            predictions[video_name] = data
            
    return predictions


def identify_noise_keyframes(predictions):
    noise_keyframes = []
    
    # Find news anchor keyframes (prediction = 1)
    news_anchor_keyframes = []
    for pred in predictions:
        if pred['prediction'] == 1:
            news_anchor_keyframes.append(pred['keyframe'])
    
    # If no news anchor keyframes found, all keyframes are noise
    if not news_anchor_keyframes:
        noise_keyframes = [pred['keyframe'] for pred in predictions]
        return noise_keyframes
    
    # Sort news anchor keyframes by frame number
    def extract_frame_number(keyframe_name):
        lesson, video, frame = parse_lesson_video_name(keyframe_name, with_frame=True)
        return int(frame)
    
    news_anchor_keyframes.sort(key=extract_frame_number)
    
    # Find first and last news anchor keyframes
    first_news_anchor = news_anchor_keyframes[0]
    last_news_anchor = news_anchor_keyframes[-1]
    
    first_frame_num = extract_frame_number(first_news_anchor)
    last_frame_num = extract_frame_number(last_news_anchor)
    
    # Identify noise keyframes
    for pred in predictions:
        keyframe_name = pred['keyframe']
        frame_num = extract_frame_number(keyframe_name)
        
        # Add to noise if:
        # 1. It's a news anchor keyframe (prediction = 1)
        # 2. It appears before the first news anchor keyframe
        # 3. It appears after the last news anchor keyframe
        if (pred['prediction'] == 1 or 
            frame_num < first_frame_num or 
            frame_num > last_frame_num):
            noise_keyframes.append(keyframe_name)
    
    return noise_keyframes


def remove_keyframe_file(keyframe_path):
    if os.path.exists(keyframe_path):
        os.remove(keyframe_path)


def process_video_keyframes(keyframes_dir, video_name, noise_keyframes):
    lesson, video = parse_lesson_video_name(video_name)
    video_dir = os.path.join(keyframes_dir, lesson, video)
    
    if not os.path.exists(video_dir):
        return
    
    keyframe_files = glob.glob(os.path.join(video_dir, "*.jpg"))
    noise_set = set(noise_keyframes)
    
    for keyframe_path in keyframe_files:
        keyframe_filename = os.path.basename(keyframe_path)
        if keyframe_filename in noise_set:
            remove_keyframe_file(keyframe_path)


def process_video(keyframes_dir, news_anchor_file, lesson_name, video_name):
    if os.path.exists(news_anchor_file):
        video_predictions = load_json(news_anchor_file)
        noise_keyframes = identify_noise_keyframes(video_predictions)
        process_video_keyframes(keyframes_dir, f"{lesson_name}_{video_name}", noise_keyframes)

def remove_noise_keyframes(keyframes_dir, news_anchor_dir, mode, lesson_name=None):
    if mode == "lesson":
        lesson_news_anchor_dir = os.path.join(news_anchor_dir, lesson_name)
        for news_anchor_file in sorted(os.listdir(lesson_news_anchor_dir)):
            if news_anchor_file.endswith("_news_anchor.json"):
                video_name = news_anchor_file.replace(f"{lesson_name}_", "").replace("_news_anchor.json", "")
                news_anchor_path = os.path.join(lesson_news_anchor_dir, news_anchor_file)
                process_video(keyframes_dir, news_anchor_path, lesson_name, video_name)
    else:
        for lesson_folder in sorted(os.listdir(news_anchor_dir)):
            lesson_news_anchor_dir = os.path.join(news_anchor_dir, lesson_folder)
            if os.path.isdir(lesson_news_anchor_dir):
                for news_anchor_file in sorted(os.listdir(lesson_news_anchor_dir)):
                    if news_anchor_file.endswith("_news_anchor.json"):
                        video_name = news_anchor_file.replace(f"{lesson_folder}_", "").replace("_news_anchor.json", "")
                        news_anchor_path = os.path.join(lesson_news_anchor_dir, news_anchor_file)
                        process_video(keyframes_dir, news_anchor_path, lesson_folder, video_name)


