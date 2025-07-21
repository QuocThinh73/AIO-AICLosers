def load_json(json_file):
    """
    Load data from a JSON file.
    Args:
        json_file (str): Path to the JSON file.
    Returns:
        dict: Data.
    """
    import json
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_video_fps(video):
    """
    Get FPS of a video.

    Args:
        video (str): Video path.

    Returns:
        float: FPS if successful, None if error.
    """
    import cv2
    
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"Cannot open video: {video}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def frame_to_seconds(frame_number, fps):
    """
    Convert frame number to seconds.

    Args:
        frame_number (int): Frame index.
        fps (float): Frames per second.

    Returns:
        float: Time in seconds.
    """
    return frame_number / fps


def seconds_to_frame(seconds, fps):
    """
    Convert seconds to frame number.

    Args:
        seconds (float): Time in seconds.
        fps (float): Frames per second.

    Returns:
        int: Frame index.
    """
    return int(seconds * fps)


def parse_lesson_video_name(name, with_frame=False):
    """
    Parse lesson, video (and optionally frame) from a name string.

    Args:
        name (str): Name string, e.g. 'L01_V001_000260.jpg' or 'L01_V001'
        with_frame (bool): If True, also parse frame (for keyframe names)

    Returns:
        tuple: (lesson (str), video (str)) or (lesson (str), video (str), frame (str))
    """
    import os
    base = os.path.basename(name)
    parts = base.split('_')
    lesson = parts[0]
    video = parts[1]
    if with_frame:
        frame = parts[2].split('.')[0]
        return lesson, video, frame
    else:
        return lesson, video


def get_lesson_directories(folder):
    """
    Scan for lesson directories in the given folder.
    Args:
        folder (str): Path to the folder containing lesson directories.
    Returns:
        list: Sorted list of (lesson_number (int), dir_name (str), dir_path (str)) tuples.
    """
    import os
    lesson_dirs = []
    if os.path.exists(folder):
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            if os.path.isdir(item_path) and item.startswith('L'):
                lesson_number = int(item[1:])
                lesson_dirs.append((lesson_number, item, item_path))
    return sorted(lesson_dirs)


def load_shot_ranges(shot_file_path):
    """
    Load shot ranges from a text file.

    Args:
        shot_file_path (str): Path to the shots text file.

    Returns:
        list: List of (start_frame (int), end_frame (int)) tuples.
    """
    shots = []
    with open(shot_file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                start, end = map(int, parts)
                shots.append((start, end))
    return shots


def find_video_file(lesson_dir_path, video_name):
    """
    Find a video file with the given name in a lesson directory.

    Args:
        lesson_dir_path (str): Directory to search in.
        video_name (str): Video name without extension.

    Returns:
        str or None: Full path to video file if found, else None.
    """
    import os
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        candidate = os.path.join(lesson_dir_path, video_name + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def get_all_keyframe_paths(keyframes_folder):
    """
    Scan for all keyframe image paths in the lesson/video directory structure.

    Args:
        keyframes_folder (str): Path to the keyframes folder (e.g. 'database/keyframes').

    Returns:
        list: Sorted list of keyframe image file paths.
    """
    import os, glob
    image_paths = []
    lesson_dirs = []
    for item in os.listdir(keyframes_folder):
        item_path = os.path.join(keyframes_folder, item)
        if os.path.isdir(item_path) and item.startswith('L'):
            lesson_dirs.append(item)
    for lesson_dir in sorted(lesson_dirs):
        lesson_path = os.path.join(keyframes_folder, lesson_dir)
        for video_item in os.listdir(lesson_path):
            video_path = os.path.join(lesson_path, video_item)
            pattern = os.path.join(video_path, "*.jpg")
            video_images = glob.glob(pattern)
            image_paths.extend(video_images)
    return sorted(image_paths)