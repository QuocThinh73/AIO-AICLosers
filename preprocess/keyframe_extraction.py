import os
import cv2
from utils.utils import load_json


def process_video(video_path, shot_path, output_keyframe_path):
    cap = None
    try:      
        os.makedirs(output_keyframe_path, exist_ok=True)
        
        if not os.path.exists(video_path):
            print(f"Lỗi: Không tìm thấy file video: {video_path}")
            return
            
        if not os.path.exists(shot_path):
            print(f"Lỗi: Không tìm thấy file shot: {shot_path}")
            return
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Lỗi: Không thể mở video file: {video_path}")
            return
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Tổng số frames: {total_frames}")
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        print(f"Đang đọc file shot JSON: {shot_path}")
        shots_data = load_json(shot_path)
        shots = shots_data["items"]
        print(f"Tổng số shots: {len(shots)}")
        
        keyframe_count = 0
        
        for i, shot in enumerate(shots):
            if i % 10 == 0:
                print(f"Xử lý shot {i}/{len(shots)}...")
                
            start_frame = shot["start_frame"]
            end_frame = shot["end_frame"]
            
            if end_frame < start_frame:
                print(f"Bỏ qua shot {i}: end_frame ({end_frame}) < start_frame ({start_frame})")
                continue
                
            shot_length = end_frame - start_frame + 1
            
            if shot_length <= 3:
                # If shot is too short, take all frames
                frame_indexes = list(range(start_frame, end_frame + 1))
            else:
                ratios = [0.15, 0.5, 0.85]
                frame_indexes = []
                for ratio in ratios:
                    frame_idx = int(start_frame + ratio * (end_frame - start_frame))
                    frame_indexes.append(frame_idx)
                    
            # Extract and save frames
            for idx in frame_indexes:
                if idx < 0 or idx >= total_frames:
                    continue
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    print(f"Không thể đọc frame {idx}")
                    continue
                
                keyframe_name = f"{video_name}_{idx:06d}.jpg"
                output_path = os.path.join(output_keyframe_path, keyframe_name)
                cv2.imwrite(output_path, frame)
                keyframe_count += 1
                
                if keyframe_count % 20 == 0:
                    print(f"Đã trích xuất {keyframe_count} keyframes...")
        
        print(f"Hoàn thành trích xuất {keyframe_count} keyframes cho video {video_name}")
                    
    except Exception as e:
        print(f"Xảy ra lỗi khi xử lý video: {str(e)}")
    finally:
        # Release video capture
        if cap is not None:
            cap.release()
            print("VideoCapture đã được giải phóng")

def extract_keyframe(input_video_dir, input_shot_dir, output_keyframe_dir, mode, lesson_name=None):
    os.makedirs(output_keyframe_dir, exist_ok=True)
    
    if mode == "lesson":
        os.makedirs(os.path.join(output_keyframe_dir, lesson_name), exist_ok=True)
        for video_file in sorted(os.listdir(os.path.join(input_video_dir, lesson_name))):
            if not video_file.endswith(".mp4"):
                continue
                
            video_name = os.path.splitext(video_file)[0]
            video_path = os.path.join(input_video_dir, lesson_name, video_file)
            
            if '_' in video_name and video_name.startswith(lesson_name):
                video_short_name = video_name[len(lesson_name)+1:]
            else:
                video_short_name = video_name
            
            shot_path = os.path.join(input_shot_dir, lesson_name, f"{video_name}_shots.json")
            output_keyframe_path = os.path.join(output_keyframe_dir, lesson_name, video_short_name)
            
            
            process_video(video_path, shot_path, output_keyframe_path)
            
    elif mode == "all":
        for lesson_folder in sorted(os.listdir(input_video_dir)):
            os.makedirs(os.path.join(output_keyframe_dir, lesson_folder), exist_ok=True)
            for video_file in sorted(os.listdir(os.path.join(input_video_dir, lesson_folder))):
                if not video_file.endswith(".mp4"):
                    continue
                    
                video_name = os.path.splitext(video_file)[0]
                video_path = os.path.join(input_video_dir, lesson_folder, video_file)
                
  
                if '_' in video_name and video_name.startswith(lesson_folder):
                    video_short_name = video_name[len(lesson_folder)+1:]
                else:
                    video_short_name = video_name

                shot_path = os.path.join(input_shot_dir, lesson_folder, f"{video_name}_shots.json")
                output_keyframe_path = os.path.join(output_keyframe_dir, lesson_folder, video_short_name)
                
                
                process_video(video_path, shot_path, output_keyframe_path)
