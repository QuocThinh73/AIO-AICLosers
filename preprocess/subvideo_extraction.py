import os
import json
import ffmpeg
from utils.utils import frame_to_seconds, get_video_fps


def extract_subvideo_clip(input_video, output_video, start_time, end_time, ffmpeg_bin):
    duration = end_time - start_time
    
    command = (
        ffmpeg
        .input(input_video, ss=start_time, t=duration)
        .output(output_video, c='copy', avoid_negative_ts='make_zero')
        .overwrite_output()
    )
    
    command.run(cmd=ffmpeg_bin)


def process_video_segments(video_path, segment_file_path, output_folder, ffmpeg_bin):
    os.makedirs(output_folder, exist_ok=True)
    
    with open(segment_file_path, 'r', encoding='utf-8') as f:
        segment_data = json.load(f)
    
    items = segment_data.get('items', [])
    
    for item in items:
        segment_id = item['id']
        start_frame = item['start_frame']
        end_frame = item['end_frame']
        
        fps = get_video_fps(video_path)
        start_time = frame_to_seconds(start_frame, fps)
        end_time = frame_to_seconds(end_frame, fps)
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{video_name}_{start_frame:06d}_{end_frame:06d}.mp4"
        output_path = os.path.join(output_folder, output_filename)
        
        extract_subvideo_clip(video_path, output_path, start_time, end_time, ffmpeg_bin)


def extract_subvideo(input_video_dir, input_segment_dir, output_subvideo_dir, mode, ffmpeg_bin, lesson_name=None):
    os.makedirs(output_subvideo_dir, exist_ok=True)
    
    if mode == "lesson":
        lesson_video_dir = os.path.join(input_video_dir, lesson_name)
        lesson_segment_dir = os.path.join(input_segment_dir, lesson_name)
        lesson_output_dir = os.path.join(output_subvideo_dir, lesson_name)
        os.makedirs(lesson_output_dir, exist_ok=True)
        
        for video_file in sorted(os.listdir(lesson_video_dir)):
            if video_file.endswith(".mp4"):
                video_path = os.path.join(lesson_video_dir, video_file)
                video_name = os.path.splitext(video_file)[0]
                segment_path = os.path.join(lesson_segment_dir, f"{video_name}_news_segment.json")
                
                if os.path.exists(segment_path):
                    video_output_dir = os.path.join(lesson_output_dir, video_name)
                    process_video_segments(video_path, segment_path, video_output_dir, ffmpeg_bin)
    
    else: 
        for lesson_folder in sorted(os.listdir(input_video_dir)):
            lesson_path = os.path.join(input_video_dir, lesson_folder)
            if os.path.isdir(lesson_path):
                lesson_segment_dir = os.path.join(input_segment_dir, lesson_folder)
                lesson_output_dir = os.path.join(output_subvideo_dir, lesson_folder)
                os.makedirs(lesson_output_dir, exist_ok=True)
                
                # Process all videos in this lesson
                for video_file in sorted(os.listdir(lesson_path)):
                    if video_file.endswith(".mp4"):
                        video_path = os.path.join(lesson_path, video_file)
                        video_name = os.path.splitext(video_file)[0]
                        segment_path = os.path.join(lesson_segment_dir, f"{video_name}_news_segment.json")
                        
                        if os.path.exists(segment_path):
                            video_output_dir = os.path.join(lesson_output_dir, video_name)
                            process_video_segments(video_path, segment_path, video_output_dir, ffmpeg_bin)