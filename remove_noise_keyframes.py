import os
import json
import glob
from typing import List, Dict, Tuple
from utils import load_json, parse_lesson_video_name, get_lesson_directories

# Configuration
KEYFRAME_FOLDER = os.path.join("database", "keyframes")
NEWS_ANCHOR_FOLDER = os.path.join("database", "news_anchor")

def load_news_anchor_predictions(news_anchor_dir: str) -> Dict[str, List[Dict[str, any]]]:
    """
    Load all news anchor prediction files.
    
    Args:
        news_anchor_dir: Path to news_anchor directory
        
    Returns:
        Dictionary mapping video names to their prediction data
    """
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


def identify_noise_keyframes(predictions: List[Dict[str, any]]) -> List[str]:
    """
    Identify noise keyframes based on news anchor predictions.
    
    Args:
        predictions: List of prediction dictionaries with 'keyframe' and 'prediction' keys
        
    Returns:
        List of keyframe filenames that should be removed
    """
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
    def extract_frame_number(keyframe_name: str) -> int:
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


def remove_keyframe_file(keyframe_path: str) -> None:
    """
    Remove a keyframe file.
    
    Args:
        keyframe_path: Full path to the keyframe file
    """
    # os.remove(keyframe_path)


def process_video_keyframes(keyframes_dir: str, video_name: str, noise_keyframes: List[str]) -> Tuple[int, int]:
    """
    Process keyframes for a specific video.
    
    Args:
        keyframes_dir: Path to keyframes directory
        video_name: Video name (e.g., 'L01_V001')
        noise_keyframes: List of noise keyframe filenames
        
    Returns:
        Tuple of (total_keyframes, removed_keyframes)
    """
    lesson, video = parse_lesson_video_name(video_name)
    video_dir = os.path.join(keyframes_dir, lesson, video)
    
    # Get all keyframe files in this video directory
    keyframe_files = glob.glob(os.path.join(video_dir, "*.jpg"))
    total_keyframes = len(keyframe_files)
    removed_count = 0
    
    # Create set of noise keyframes for faster lookup
    noise_set = set(noise_keyframes)
    
    # Remove noise keyframes
    for keyframe_path in keyframe_files:
        keyframe_filename = os.path.basename(keyframe_path)
        if keyframe_filename in noise_set:
            remove_keyframe_file(keyframe_path)
            removed_count += 1
    
    return total_keyframes, removed_count


def main():
    predictions = load_news_anchor_predictions(NEWS_ANCHOR_FOLDER)
    
    total_processed = 0
    total_removed = 0
    
    # Process each video
    for video_name, video_predictions in predictions.items():
        # Identify noise keyframes for this video
        noise_keyframes = identify_noise_keyframes(video_predictions)
        
        # Remove noise keyframes
        total_keyframes, removed_count = process_video_keyframes(KEYFRAME_FOLDER, video_name, noise_keyframes)
        
        total_processed += total_keyframes
        total_removed += removed_count
    
    print(f"Total keyframes removed: {total_removed}")
    print(f"Removal rate: {total_removed/total_processed*100:.2f}%" if total_processed > 0 else "No keyframes processed")


if __name__ == "__main__":
    main() 