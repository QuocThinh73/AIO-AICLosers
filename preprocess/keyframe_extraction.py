import os
import cv2
import json
import glob

# Configuration
N_KEYFRAMES = 3

def extract_keyframes_from_shot(video_path, shot_ranges, output_dir, n_keyframes=N_KEYFRAMES):
    """
    Extract keyframes from video shots and save them in the output directory
    
    Args:
        video_path: Path to the video file
        shot_ranges: List of (start_frame, end_frame) tuples
        output_dir: Directory to save keyframes
        n_keyframes: Number of keyframes to extract per shot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
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
            
            # Save with format based on the video name and frame index
            filename = f"{video_name}_{idx:06d}.jpg"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, frame)
            
    cap.release()

def load_shot_ranges(shot_file_path):
    """
    Load shot ranges from a shot file
    
    Args:
        shot_file_path: Path to shot file
    
    Returns:
        List of (start_frame, end_frame) tuples
    """
    if shot_file_path.endswith('.json'):
        with open(shot_file_path, 'r') as f:
            data = json.load(f)
            shots = [(item["start_frame"], item["end_frame"]) for item in data["items"]]
    else:  # Assume it's a txt file
        shots = []
        with open(shot_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    start, end = int(parts[0]), int(parts[1])
                    shots.append((start, end))
    return shots

def extract_keyframe(input_video_dir, input_shot_dir, output_keyframe_dir, mode, lesson_name=None):
    os.makedirs(output_keyframe_dir, exist_ok=True)
    
    if mode == "lesson":
        if not lesson_name:
            raise ValueError("Lesson name is required when mode is 'lesson'")
        
        # Create output directory for this lesson
        lesson_output_dir = os.path.join(output_keyframe_dir, lesson_name)
        os.makedirs(lesson_output_dir, exist_ok=True)
        
        # Process all videos in this lesson
        lesson_shot_dir = os.path.join(input_shot_dir, lesson_name)
        lesson_video_dir = os.path.join(input_video_dir, lesson_name)
        
        if not os.path.exists(lesson_shot_dir) or not os.path.exists(lesson_video_dir):
            raise ValueError(f"Shot or video directory for lesson {lesson_name} does not exist")
        
        for shot_file in sorted(os.listdir(lesson_shot_dir)):
            # Skip non-shot files
            if not (shot_file.endswith('.txt') or shot_file.endswith('.json')):
                continue
                
            # Get video name from shot file name
            video_name = os.path.splitext(shot_file)[0]
            if video_name.endswith('_shots'):
                video_name = video_name[:-6]  # Remove '_shots' suffix
            
            # Find the video file
            video_file = None
            for ext in ['.mp4', '.avi', '.mkv']:
                potential_video = os.path.join(lesson_video_dir, video_name + ext)
                if os.path.exists(potential_video):
                    video_file = potential_video
                    break
            
            if not video_file:
                print(f"Video file for {video_name} not found. Skipping...")
                continue
            
            # Create output directory for this video
            video_output_dir = os.path.join(lesson_output_dir, video_name)
            os.makedirs(video_output_dir, exist_ok=True)
            
            # Load shot ranges
            shot_file_path = os.path.join(lesson_shot_dir, shot_file)
            shots = load_shot_ranges(shot_file_path)
            
            # Extract keyframes
            extract_keyframes_from_shot(video_file, shots, video_output_dir)
            
    else:  # mode == "all"
        # Process all lessons
        for lesson_name in sorted(os.listdir(input_video_dir)):
            lesson_shot_dir = os.path.join(input_shot_dir, lesson_name)
            lesson_video_dir = os.path.join(input_video_dir, lesson_name)
            lesson_output_dir = os.path.join(output_keyframe_dir, lesson_name)
            
            if not os.path.exists(lesson_shot_dir) or not os.path.exists(lesson_video_dir):
                continue
                
            os.makedirs(lesson_output_dir, exist_ok=True)
            
            for shot_file in sorted(os.listdir(lesson_shot_dir)):
                # Skip non-shot files
                if not (shot_file.endswith('.txt') or shot_file.endswith('.json')):
                    continue
                    
                # Get video name from shot file name
                video_name = os.path.splitext(shot_file)[0]
                if video_name.endswith('_shots'):
                    video_name = video_name[:-6]  # Remove '_shots' suffix
                
                # Find the video file
                video_file = None
                for ext in ['.mp4', '.avi', '.mkv']:
                    potential_video = os.path.join(lesson_video_dir, video_name + ext)
                    if os.path.exists(potential_video):
                        video_file = potential_video
                        break
                
                if not video_file:
                    print(f"Video file for {video_name} not found. Skipping...")
                    continue
                
                # Create output directory for this video
                video_output_dir = os.path.join(lesson_output_dir, video_name)
                os.makedirs(video_output_dir, exist_ok=True)
                
                # Load shot ranges
                shot_file_path = os.path.join(lesson_shot_dir, shot_file)
                shots = load_shot_ranges(shot_file_path)
                
                # Extract keyframes
                extract_keyframes_from_shot(video_file, shots, video_output_dir)
