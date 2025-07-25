import os
from pathlib import Path
from utils import frame_to_seconds, get_video_fps
import ffmpeg

# ===== CONFIG =====
VIDEOS_FOLDER = os.path.join("database", "videos")
SEGMENT_FOLDER = os.path.join("database", "news_segment")
SUBVIDEO_FOLDER = os.path.join("database", "subvideos")
FFMPEG_BIN = r"D:\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"

def extract_subvideo(input_video, output_video, start_time, end_time):
    """
    Extract a subvideo using FFmpeg.
    Args:
        input_video (str): Path to input video file.
        output_video (str): Path to output video file.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
    """
    duration = end_time - start_time
    (
        ffmpeg
        .input(input_video, ss=start_time, t=duration)
        .output(output_video, c='copy', avoid_negative_ts='make_zero')
        .overwrite_output()
        .run(cmd=FFMPEG_BIN)
    )


def process_video_segments(video_path, segment_data, output_folder):
    """
    Process segments for a single video and create subvideos.
    Args:
        video_path (str): Path to the input video file.
        segment_data (dict): Segment data with total and items.
        output_folder (str): Path to output folder for subvideos.
    Returns:
        int: Number of successfully created subvideos.
    """
    items = segment_data.get('items', [])
    
    for item in items:
        segment_id = item['id']
        start_frame = item['start_frame']
        end_frame = item['end_frame']
        
        # Convert frames to time
        fps = get_video_fps(video_path)
        start_time = frame_to_seconds(start_frame, fps)
        end_time = frame_to_seconds(end_frame, fps)
        
        # Create output filename
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{video_name}_{start_frame:06d}_{end_frame:06d}.mp4"
        output_path = os.path.join(output_folder, output_filename)
        
        # Extract subvideo
        extract_subvideo(video_path, output_path, start_time, end_time)


def segment_all_subvideos(videos_folder, segment_folder, output_folder):
    """
    Segment all videos into subvideos based on segment data.
    Args:
        videos_folder (str): Path to the videos folder.
        segment_folder (str): Path to the segment data folder.
        output_folder (str): Path to the output folder for subvideos.
    """
    videos_path = Path(videos_folder)
    segment_path = Path(segment_folder)
    output_path = Path(output_folder)
    
    # Process each lesson folder
    for lesson_path in videos_path.iterdir():
        if lesson_path.is_dir():
            lesson_name = lesson_path.name
            
            # Create lesson output folder
            lesson_output_path = output_path / lesson_name
            lesson_output_path.mkdir(exist_ok=True)
            
            # Find corresponding segment folder
            segment_lesson_path = segment_path / lesson_name
            
            # Process each video in the lesson
            for video_file in lesson_path.glob("*.mp4"):
                video_name = video_file.stem
                
                # Find corresponding segment file
                segment_file = segment_lesson_path / f"{video_name}_news_segment.txt"
                
                # Create video output folder
                video_output_path = lesson_output_path / video_name
                video_output_path.mkdir(exist_ok=True)
                
                # Load segment data from txt
                segment_data = load_segment_txt(str(segment_file))
                
                # Process video segments
                process_video_segments(str(video_file), segment_data, str(video_output_path))


def load_segment_txt(txt_file):
    """
    Đọc file txt segment và chuyển thành dict phù hợp cho process_video_segments.
    """
    items = []
    with open(txt_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            start_frame, end_frame = map(int, parts)
            items.append({
                'id': idx + 1,
                'start_frame': start_frame,
                'end_frame': end_frame
            })
    return {'items': items}


def main():
    os.makedirs(SUBVIDEO_FOLDER, exist_ok=True)
    segment_all_subvideos(VIDEOS_FOLDER, SEGMENT_FOLDER, SUBVIDEO_FOLDER)


if __name__ == "__main__":
    main() 