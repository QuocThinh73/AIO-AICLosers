# %% [markdown]
# # Object Detection với Grounding DINO
# 
# Notebook này sử dụng Grounding DINO để thực hiện object detection trên các keyframe từ video, sử dụng caption đã được trích xuất trước đó.
# 
# ## 1. Cài đặt thư viện

# %%
# Cài đặt các thư viện cần thiết
!pip install -q torch torchvision
!pip install -q transformers
!pip install -q timm
!pip install -q huggingface_hub
!pip install -q opencv-python-headless
!pip install -q gdown
!pip install -q matplotlib
!pip install -q Pillow
!pip install -q groundingdino-py

# %% [markdown]
# ## 2. Tải dữ liệu từ Google Drive

# %%
# Cấu hình batch và ID Google Drive
BATCH_NAME = "L01"  # Tên batch (L01, L02, L03, ...)
BATCH_ID = "14MeYV2WBWwldMDGRrpG9s7vz8triwbWr"  # ID cho L01.zip
BATCH_RESULT_ID = "15AVPGtZ6W3C3H8Hc_JF3SBsrhMVUbZWU"  # ID của file results

# Tạo thư mục data nếu chưa có
!mkdir -p data

# Tải file batch (keyframes) từ Google Drive
print(f"Tải xuống batch {BATCH_NAME}...")
!gdown {BATCH_ID} -O data/{BATCH_NAME}.zip
!unzip -qq data/{BATCH_NAME}.zip -d ./

# Tải file kết quả caption từ Google Drive
print(f"Tải xuống kết quả caption...")
!gdown {BATCH_RESULT_ID} -O data/results.zip
!unzip -qq data/results.zip -d data/

print("Đã tải xong dữ liệu!")

# %% [markdown]
# ## 3. Import thư viện

# %%
import torch
import json
import glob
import os
import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from PIL import Image
import numpy as np

# Xử lý kết nối Google Drive nếu chạy trong Colab
try:
    from google.colab import drive
    try:
        drive.mount('/content/drive')
        print("[INFO] Đã kết nối Google Drive qua Colab")
    except NotImplementedError:
        print("[INFO] Môi trường hiện tại không hỗ trợ mount Google Drive theo cách của Colab")
        # Trong Kaggle, dữ liệu được truy cập trực tiếp từ thư mục hiện tại
except ImportError:
    print("[INFO] Không chạy trong Colab, bỏ qua mount Google Drive")

# Imports cho Grounding DINO sẽ được thêm vào phần tải model

# %% [markdown]
# ## 4. Kiểm tra GPU

# %%
# CUDA diagnosis section
print("\nGPU Diagnostic Information:")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU 0 name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Check NVIDIA drivers and PyTorch CUDA installation.")

# %% [markdown]
# ## 5. Cấu hình batch và chỉ số video

# %%
# Cấu hình xử lý batch
START_VIDEO_INDEX = 1  # Bắt đầu từ V001
BATCH_SIZE = 2  # Xử lý 2 video một lần

# Định nghĩa đường dẫn
BATCH_PATH = BATCH_NAME  # Ví dụ: "L01"

# Lấy danh sách video trong batch
videos = sorted(glob.glob(os.path.join(BATCH_PATH, "V*")))
print(f"Tìm thấy {len(videos)} thư mục video trong batch {BATCH_PATH}")

# Chỉ xử lý các video từ START_VIDEO_INDEX đến START_VIDEO_INDEX + BATCH_SIZE - 1
end_idx = min(START_VIDEO_INDEX + BATCH_SIZE - 1, len(videos))
selected_videos = videos[START_VIDEO_INDEX - 1:end_idx]
print(f"Xử lý {len(selected_videos)} video: {[os.path.basename(v) for v in selected_videos]}")

# %% [markdown]
# ## 6. Tải model Grounding DINO

# %%
# Tải model Grounding DINO
device = "cuda" if torch.cuda.is_available() else "cpu"

# Sử dụng groundingdino-py trực tiếp
from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T

# Đường dẫn cho model config và checkpoint
model_config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
model_checkpoint_path = "weights/groundingdino_swint_ogc.pth"

# Tải cấu hình và checkpoint
!mkdir -p GroundingDINO/groundingdino/config
!mkdir -p weights
!wget -q -O {model_config_path} https://github.com/IDEA-Research/GroundingDINO/raw/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
!wget -q -O {model_checkpoint_path} https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Tải model
model = load_model(model_config_path, model_checkpoint_path)
model.to(device)

print(f"Đã tải xong model Grounding DINO trên device: {device}")

# %% [markdown]
# ## 7. Hàm hỗ trợ để trích xuất đối tượng từ caption

# %%
# Hàm trích xuất các đoạn có ý nghĩa từ caption để làm prompt
def extract_objects_from_caption(caption):
    """
    Trích xuất các đoạn có ý nghĩa từ caption để dùng làm prompt cho object detection
    """
    # Cắt bỏ các phần meta không cần thiết
    caption = caption.replace("The image appears to be", "")
    caption = caption.replace("The image shows", "")
    
    # Danh sách để lưu các đoạn prompt
    prompts = []
    
    # Chia caption thành các phần theo dấu ngắt dòng
    parts = caption.split('\n')
    
    for part in parts:
        # Bỏ qua các dòng ngắn hoặc không có nội dung
        if len(part.strip()) < 10:
            continue
            
        # Tìm các phần mô tả cụ thể
        if ':' in part and '**' in part:
            # Ví dụ: "**Top Row:** - People walking..."
            topic_parts = part.split(':')
            if len(topic_parts) > 1 and len(topic_parts[1].strip()) > 10:
                prompts.append(topic_parts[1].strip())
        elif '-' in part:
            # Chia thành các phần theo dấu gạch ngang
            bullet_points = part.split('-')
            for point in bullet_points:
                if len(point.strip()) > 10:
                    prompts.append(point.strip())
        elif len(part.strip()) > 20 and part.strip().endswith('.'):
            # Lấy các câu hoàn chỉnh
            sentences = part.split('.')
            for sentence in sentences:
                if len(sentence.strip()) > 20:
                    prompts.append(sentence.strip() + '.')
    
    # Nếu không tìm được prompt nào, dùng caption gốc
    if not prompts:
        # Nếu caption quá dài, chia thành các đoạn nhỏ hơn
        if len(caption) > 200:
            sentences = caption.split('.')
            for sentence in sentences:
                if len(sentence.strip()) > 20:
                    prompts.append(sentence.strip() + '.')
        else:
            prompts.append(caption)
    
    # Giới hạn số lượng prompt để không quá nặng
    return prompts[:3]

# Hàm phát hiện đối tượng với Grounding DINO
def detect_objects(image_path, object_name, box_threshold=0.35, text_threshold=0.25):
    try:
        # Tải và tiền xử lý ảnh
        image_source, image = load_image(image_path)
        
        # Phát hiện đối tượng
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=f"Find {object_name}",
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device
        )
        
        # Chuyển đổi sang định dạng thông thường
        H, W, _ = image_source.shape
        boxes_xyxy = boxes * torch.Tensor([W, H, W, H])
        boxes_xyxy = boxes_xyxy.cpu().numpy().tolist()
        
        return {
            "boxes": boxes_xyxy,
            "scores": logits.cpu().numpy().tolist(),
            "labels": phrases
        }
    except Exception as e:
        print(f"Lỗi khi xử lý {os.path.basename(image_path)} với đối tượng '{object_name}': {e}")
        return {
            "boxes": [],
            "scores": [],
            "labels": []
        }

# Hàm vẽ kết quả lên ảnh
def visualize_detection(image_path, detection_results):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for box, score, label in zip(detection_results["boxes"], detection_results["scores"], detection_results["labels"]):
        x1, y1, x2, y2 = map(int, box)
        
        # Vẽ bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Vẽ nhãn và điểm số
        text = f"{label}: {score:.2f}"
        cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    return image

# %% [markdown]
# ## 8. Xử lý và phát hiện đối tượng

# %%
# Tạo thư mục lưu kết quả
os.makedirs("detection_results", exist_ok=True)

# Xử lý từng video
for video_dir in tqdm(selected_videos, desc="Xử lý video"):
    video_name = os.path.basename(video_dir)
    
    # Tìm file caption tương ứng
    caption_file = os.path.join("data", "results", f"{BATCH_NAME}_{video_name}_caption.json")
    
    # Kiểm tra nếu file caption tồn tại
    if not os.path.exists(caption_file):
        print(f"[Lỗi] Không tìm thấy file caption cho video {video_name}: {caption_file}")
        continue
    
    # Đọc file caption
    with open(caption_file, 'r', encoding='utf-8') as f:
        captions = json.load(f)
    
    # Khởi tạo danh sách lưu kết quả
    detection_results = []
    
    # Xử lý từng keyframe
    for item in tqdm(captions, desc=video_name):
        keyframe_name = item["keyframe"]
        caption = item["caption"]
        
        # Đường dẫn đầy đủ đến file keyframe
        keyframe_path = os.path.join(video_dir, keyframe_name)
        
        # Trích xuất các đoạn prompt từ caption
        prompts = extract_objects_from_caption(caption)
        
        # Khởi tạo kết quả cho keyframe hiện tại
        keyframe_results = {
            "keyframe": keyframe_name,
            "caption": caption,
            "objects": []
        }
        
        # Phát hiện với từng prompt
        for prompt in prompts:
            # Phát hiện đối tượng
            results = detect_objects(keyframe_path, prompt)
            
            # Thêm kết quả vào danh sách
            for i, (box, score) in enumerate(zip(results["boxes"], results["scores"])):
                label = results["labels"][i] if i < len(results["labels"]) else prompt
                keyframe_results["objects"].append({
                    "prompt": prompt,  # Lưu lại prompt đã sử dụng
                    "object": label,
                    "box": box,
                    "score": score
                })
        
        # Thêm kết quả của keyframe vào danh sách chung
        detection_results.append(keyframe_results)
    
    # Lưu kết quả phát hiện vào file JSON
    output_file = os.path.join("detection_results", f"{BATCH_NAME}_{video_name}_detection.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detection_results, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Đã lưu kết quả phát hiện cho video {video_name} vào file {output_file}")

# %% [markdown]
# ## 9. Nén kết quả để tải về

# %%
# Nén thư mục kết quả để tải về
!zip -r detection_results.zip detection_results

print("✅ Đã nén kết quả vào file detection_results.zip")
print("👉 Bạn có thể tải file này về từ panel Files ở bên trái.")
