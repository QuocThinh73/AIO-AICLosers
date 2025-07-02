import os
import cv2
import numpy as np

# Configuration
N_KEYFRAMES = 3
SHOT_FOLDER = os.path.join("database", "shots")
VIDEO_FOLDER = os.path.join("database", "videos")
KEYFRAME_FOLDER = os.path.join("database", "keyframes")

os.makedirs(KEYFRAME_FOLDER, exist_ok=True)

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
    output_dir = os.path.join(KEYFRAME_FOLDER, f"L{l_batch:02d}", f"V{v_number:03d}")
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
            # Use custom ratios: 0.15, 0.5, 0.85 instead of even distribution
            ratios = [0.15, 0.5, 0.85][:n_keyframes]  # Take only needed ratios
            frame_indexes = []
            for ratio in ratios:
                frame_idx = int(start + ratio * (end - start))
                frame_indexes.append(frame_idx)

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

def main():
    from utils import parse_lesson_video_name, get_lesson_directories, load_shot_ranges, find_video_file
    lesson_dirs = get_lesson_directories(VIDEO_FOLDER)
    
    # Process each L batch
    for lesson_number, lesson_dir_name, lesson_dir_path in lesson_dirs:
        if not os.path.exists(lesson_dir_path):
            continue
        
        # Process all shot files
        for filename in os.listdir(SHOT_FOLDER):
            if not filename.endswith('.txt'):
                continue
                
            video_name = filename.replace("_shots.txt", "")
            result = parse_lesson_video_name(video_name, with_frame=False)
            parsed_lesson, parsed_video = result[0], result[1]
            parsed_lesson_number = int(parsed_lesson[1:])
            parsed_video_number = int(parsed_video[1:])
            
            # Only process videos that belong to current L batch
            if parsed_lesson_number != lesson_number:
                continue
            
            # Load shot ranges
            shot_file_path = os.path.join(SHOT_FOLDER, filename)
            shots = load_shot_ranges(shot_file_path)
            
            # Find corresponding video file
            video_path = find_video_file(lesson_dir_path, video_name)
            if not video_path:
                continue
            
            # Extract keyframes
            extract_keyframes_from_shot(video_path, shots, lesson_number, parsed_video_number, N_KEYFRAMES)

if __name__ == "__main__":
    main()
