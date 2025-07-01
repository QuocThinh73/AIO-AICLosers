import os
import json
from pathlib import Path
from utils import extract_keyframe_info

# ===== CONFIG CONSTANTS =====
INPUT_DIR = os.path.join("database", "news_anchor")
OUTPUT_DIR = os.path.join("database", "news_segment")
FPS = 30


def load_classification_results(json_file):
    """
    Load kết quả phân loại từ file JSON
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    data.sort(key=lambda x: int(extract_keyframe_info(x['keyframe'])[2]))
    return data


def segment_news_from_file(json_file):
    """
    Phân đoạn tin tức từ một file kết quả phân loại
    """
    data = load_classification_results(json_file)
    segments = []
    current_start_frame = None
    prev_prediction = 0  # Trạng thái trước đó
    
    for frame_data in data:
        keyframe = frame_data['keyframe']
        prediction = frame_data['prediction']
        _, _, frame_str = extract_keyframe_info(keyframe)
        frame_number = int(frame_str)
        
        # Chỉ bắt đầu tin tức mới khi chuyển từ 0 -> 1 (lần đầu gặp người dẫn chương trình)
        if prediction == 1 and prev_prediction == 0:
            # Kết thúc segment hiện tại nếu có
            if current_start_frame is not None:
                segments.append((current_start_frame, prev_frame_number))
            
            # Bắt đầu segment mới
            current_start_frame = frame_number
        
        prev_prediction = prediction
        prev_frame_number = frame_number
    
    # Kết thúc segment cuối cùng
    if current_start_frame is not None:
        segments.append((current_start_frame, prev_frame_number))
    
    return segments


def segment_all_videos(input_dir, output_dir):
    """
    Phân đoạn tin tức cho tất cả video trong tất cả batch
    """
    input_path = Path(input_dir)
    
    # Duyệt qua tất cả thư mục batch
    for batch_dir in input_path.iterdir():
        if batch_dir.is_dir():
            batch_name = batch_dir.name
            
            # Tạo thư mục output cho batch
            batch_output_path = Path(output_dir) / batch_name
            batch_output_path.mkdir(parents=True, exist_ok=True)
            
            # Duyệt qua tất cả file JSON trong batch
            for json_file in batch_dir.glob("*.json"):
                segments = segment_news_from_file(str(json_file))
                video_id = json_file.stem.replace('_news_anchor', '')
                
                # Lưu kết quả theo format giống database/shots
                output_file = batch_output_path / f"{video_id}_news_segment.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for start_frame, end_frame in segments:
                        f.write(f"{start_frame} {end_frame}\n")

if __name__ == "__main__":
    segment_all_videos(INPUT_DIR, OUTPUT_DIR)