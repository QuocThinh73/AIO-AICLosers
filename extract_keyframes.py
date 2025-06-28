import os
import cv2
import numpy as np
from pathlib import Path

# Configuration
N_KEYFRAMES = 3
SHOT_TXT_FOLDER = os.path.join("database", "shots")
VIDEO_BASE_FOLDER = os.path.join("database", "videos")
OUTPUT_FOLDER = os.path.join("database", "keyframes")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def extract_keyframes_from_shot(video_path, shot_ranges, l_batch, v_number, n_keyframes):
    """
    Extract keyframes from video shots and save with format: L01_V001_000001.jpg
    
    Args:
        video_path: Path to the video file
        shot_ranges: List of (start_frame, end_frame) tuples
        l_batch: L batch number (e.g., 1 for L01)
        v_number: V video number (e.g., 1 for V001)
        n_keyframes: Number of keyframes to extract per shot
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directory: database/keyframes/L01/V001/
    output_dir = os.path.join(OUTPUT_FOLDER, f"L{l_batch:02d}", f"V{v_number:03d}")
    os.makedirs(output_dir, exist_ok=True)

    for start, end in shot_ranges:
        if end < start:
            continue
            
        shot_length = end - start + 1
        
        # Determine frame indices to extract
        if shot_length <= n_keyframes:
            # If shot is too short, take all frames
            frame_indexes = list(range(start, end + 1))
        else:
            # Evenly distribute n_keyframes across the shot
            frame_indexes = np.linspace(start, end, n_keyframes, dtype=int).tolist()

        # Extract and save frames
        for idx in frame_indexes:
            if idx < 0 or idx >= total_frames:
                continue
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Save with format: L01_V001_000001.jpg
            filename = f"L{l_batch:02d}_V{v_number:03d}_{idx:06d}.jpg"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, frame)
            
    cap.release()

def parse_video_name(video_name):
    """
    Parse video filename to extract L batch and V number.
    
    Expected formats:
    - L01_V001 -> (1, 1)
    - V001 -> (1, 1) [defaults to L01]
    
    Args:
        video_name: Video filename without extension
        
    Returns:
        tuple: (l_batch, v_number) as integers
    """
    # Default fallback values
    l_batch, v_number = 1, 1
    
    # Format: L01_V001
    parts = video_name.split('_')
    l_part = parts[0]  # L01
    v_part = parts[1]  # V001

    # Extract numeric values
    l_batch = int(l_part[1:])  # L01 -> 1
    v_number = int(v_part[1:])  # V001 -> 1
    
    return l_batch, v_number

def get_l_batch_directories():
    """
    Scan for L batch directories in the video base folder.
    
    Returns:
        list: Sorted list of (l_num, dir_name, dir_path) tuples
    """
    l_batches = []
    
    if os.path.exists(VIDEO_BASE_FOLDER):
        for item in os.listdir(VIDEO_BASE_FOLDER):
            item_path = os.path.join(VIDEO_BASE_FOLDER, item)
            if os.path.isdir(item_path) and item.startswith('L'):
                l_num = int(item[1:])  # L01 -> 1
                l_batches.append((l_num, item, item_path))
    
    return sorted(l_batches)

def load_shot_ranges(shot_file_path):
    """
    Load shot ranges from a text file.
    
    Args:
        shot_file_path: Path to the shots text file
        
    Returns:
        list: List of (start_frame, end_frame) tuples
    """
    shots = []
    with open(shot_file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                start, end = map(int, parts)
                shots.append((start, end))
    return shots

def find_video_file(l_dir_path, video_name):
    """
    Find video file with given name in directory.
    
    Args:
        l_dir_path: Directory to search in
        video_name: Video name without extension
        
    Returns:
        str or None: Full path to video file if found
    """
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        candidate = os.path.join(l_dir_path, video_name + ext)
        if os.path.isfile(candidate):
            return candidate
    return None

def main():
    """
    Main processing function that extracts keyframes from all videos.
    """
    l_batches = get_l_batch_directories()
    
    # Process each L batch
    for l_num, l_dir_name, l_dir_path in l_batches:
        if not os.path.exists(l_dir_path):
            continue
        
        # Process all shot files
        for filename in os.listdir(SHOT_TXT_FOLDER):
            if not filename.endswith('.txt'):
                continue
                
            video_name = filename.replace("_shots.txt", "")
            parsed_l, parsed_v = parse_video_name(video_name)
            
            # Only process videos that belong to current L batch
            if parsed_l != l_num:
                continue
            
            # Load shot ranges
            shot_file_path = os.path.join(SHOT_TXT_FOLDER, filename)
            shots = load_shot_ranges(shot_file_path)
            
            # Find corresponding video file
            video_path = find_video_file(l_dir_path, video_name)
            if not video_path:
                continue
            
            # Extract keyframes
            extract_keyframes_from_shot(video_path, shots, l_num, parsed_v, N_KEYFRAMES)

if __name__ == "__main__":
    main()
