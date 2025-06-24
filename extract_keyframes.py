import os
import cv2
import numpy as np

# ==============================
N_KEYFRAMES = 3
SHOT_TXT_FOLDER = os.path.join("data", "shots")
VIDEO_FOLDER = os.path.join("data", "videos", "L001")
OUTPUT_FOLDER = os.path.join("data", "keyframes")
# ==============================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def extract_keyframes_from_shot(video_path, shot_ranges, video_name, n_keyframes):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for start, end in shot_ranges:
        print(f"Processing shot {start} to {end}")
        frame_indexes = []
        if end < start:
            continue
        shot_length = end - start + 1
        if shot_length <= n_keyframes:
            # Nếu shot ngắn quá, lấy toàn bộ frame trong shot
            frame_indexes = list(range(start, end+1))
        else:
            # Chia đều ra n_keyframes frame trong shot (bao gồm cả start, end)
            frame_indexes = np.linspace(start, end, n_keyframes, dtype=int).tolist()

        for idx in frame_indexes:
            if idx < 0 or idx >= total_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            # Tên file: {video_name}_{frame_index gốc}.jpg
            out_name = f"{video_name}_{idx}.jpg"
            out_path = os.path.join(OUTPUT_FOLDER, out_name)
            cv2.imwrite(out_path, frame)
    cap.release()

def main():
    # Liệt kê tất cả file .txt trong folder shot_results
    for fname in os.listdir(SHOT_TXT_FOLDER):
        if fname.endswith('.txt'):
            video_name = fname.replace("_shots.txt", "")
            shot_file_path = os.path.join(SHOT_TXT_FOLDER, fname)
            # Đọc shot start-end từ file txt
            shots = []
            with open(shot_file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        start, end = map(int, parts)
                        shots.append((start, end))
            # Tìm file video tương ứng
            video_path = None
            for ext in ['.mp4', '.avi', '.mov3', '.mkv']:
                candidate = os.path.join(VIDEO_FOLDER, video_name + ext)
                if os.path.isfile(candidate):
                    video_path = candidate
                    break
            if not video_path:
                print(f"Không tìm thấy video tương ứng cho {fname}")
                continue
            print(f"Processing {video_path} ...")
            extract_keyframes_from_shot(video_path, shots, video_name, N_KEYFRAMES)
    print("DONE! Các keyframe đã lưu trong:", OUTPUT_FOLDER)

if __name__ == "__main__":
    main()
