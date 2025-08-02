import os
import json
from pathlib import Path
import sys
from .utils import parse_lesson_video_name


def load_classification_results(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def get_frame_number_from_keyframe(keyframe):
        result = parse_lesson_video_name(keyframe, with_frame=True)
        if len(result) == 3:
            frame_str = result[2]
            return int(frame_str)
        raise ValueError(f"Keyframe name {keyframe} does not have frame info.")

    data.sort(key=lambda x: get_frame_number_from_keyframe(x['keyframe']))
    return data


def segment_news_from_file(json_file):
    data = load_classification_results(json_file)
    segments = []
    current_start_frame = None
    prev_prediction = 0
    prev_frame_number = None

    for frame_data in data:
        keyframe = frame_data['keyframe']
        prediction = frame_data['prediction']
        result = parse_lesson_video_name(keyframe, with_frame=True)
        if len(result) == 3:
            frame_str = result[2]
            frame_number = int(frame_str)
        else:
            raise ValueError(f"Keyframe name {keyframe} does not have frame info.")

        # Start a new segment when prediction changes from 0 to 1 (first anchor appearance)
        if prediction == 1 and prev_prediction == 0:
            # End the current segment if it exists
            if current_start_frame is not None:
                segments.append((current_start_frame, prev_frame_number))
            # Start a new segment
            current_start_frame = frame_number

        prev_prediction = prediction
        prev_frame_number = frame_number

    # End the last segment
    if current_start_frame is not None:
        segments.append((current_start_frame, prev_frame_number))

    # Convert segments to JSON format
    items = []
    for i, (start_frame, end_frame) in enumerate(segments):
        items.append({
            "id": i,
            "start_frame": start_frame,
            "end_frame": end_frame,
        })
    
    data = {
        "total": len(items),
        "items": items
    }
    
    return data


def process_video(news_anchor_file, output_segment_path):
    # Create output directory if not exists
    os.makedirs(os.path.dirname(output_segment_path), exist_ok=True)
    
    # Segment news from news anchor file
    segment_data = segment_news_from_file(news_anchor_file)
    
    # Save the segment data to file
    with open(output_segment_path, 'w', encoding='utf-8') as f:
        json.dump(segment_data, f, indent=4, ensure_ascii=False)


def segment_news(input_keyframe_dir, input_news_anchor_dir, output_news_segment_dir, mode, lesson_name=None):
    os.makedirs(output_news_segment_dir, exist_ok=True)
    
    if mode == "lesson":
        lesson_news_anchor_dir = os.path.join(input_news_anchor_dir, lesson_name)
        lesson_output_dir = os.path.join(output_news_segment_dir, lesson_name)
        os.makedirs(lesson_output_dir, exist_ok=True)
        
        # Process all news anchor files in the lesson directory
        for news_anchor_file in sorted(os.listdir(lesson_news_anchor_dir)):
            if news_anchor_file.endswith("_news_anchor.json"):
                input_file_path = os.path.join(lesson_news_anchor_dir, news_anchor_file)
                video_id = news_anchor_file.replace('_news_anchor.json', '')
                output_file_path = os.path.join(lesson_output_dir, f"{video_id}_news_segment.json")
                process_video(input_file_path, output_file_path)
    
    else:  
        # Process all lessons
        for lesson_folder in sorted(os.listdir(input_news_anchor_dir)):
            lesson_path = os.path.join(input_news_anchor_dir, lesson_folder)
            if os.path.isdir(lesson_path):
                lesson_output_dir = os.path.join(output_news_segment_dir, lesson_folder)
                os.makedirs(lesson_output_dir, exist_ok=True)
                
                # Process all news anchor files in this lesson
                for news_anchor_file in sorted(os.listdir(lesson_path)):
                    if news_anchor_file.endswith("_news_anchor.json"):
                        input_file_path = os.path.join(lesson_path, news_anchor_file)
                        video_id = news_anchor_file.replace('_news_anchor.json', '')
                        output_file_path = os.path.join(lesson_output_dir, f"{video_id}_news_segment.json")
                        process_video(input_file_path, output_file_path)

