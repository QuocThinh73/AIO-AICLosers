import os
import json
from pathlib import Path
from utils import parse_lesson_video_name

# ===== CONFIG =====
INPUT_FOLDER = os.path.join("database", "news_anchor")
OUTPUT_FOLDER = os.path.join("database", "news_segment")


def load_classification_results(json_file):
    """
    Load classification results from a JSON file and sort by frame number.
    Args:
        json_file (str): Path to the JSON file.
    Returns:
        list: Sorted list of classification results.
    """
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
    """
    Segment news from a classification result file.
    Args:
        json_file (str): Path to the classification result JSON file.
    Returns:
        list: List of (start_frame, end_frame) tuples for each news segment.
    """
    data = load_classification_results(json_file)
    segments = []
    current_start_frame = None
    prev_prediction = 0  # Previous prediction state
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

    return segments


def segment_all_videos(input_folder, output_folder):
    """
    Segment news for all videos in all batches.
    Args:
        input_folder (str or Path): Path to the input folder containing batch directories.
        output_folder (str or Path): Path to the output folder for segments.
    """
    input_path = Path(input_folder)
    for batch_path in input_path.iterdir():
        if batch_path.is_dir():
            batch_name = batch_path.name
            batch_output_path = Path(output_folder) / batch_name
            batch_output_path.mkdir(parents=True, exist_ok=True)
            for json_path in batch_path.glob("*.json"):
                segments = segment_news_from_file(str(json_path))
                video_id = json_path.stem.replace('_news_anchor', '')
                
                # Create JSON structure
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
                
                output_file = batch_output_path / f"{video_id}_news_segment.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    segment_all_videos(INPUT_FOLDER, OUTPUT_FOLDER)