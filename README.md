# HCMAI2025 - Video Preprocessing Pipeline

## Tổng quan

Pipeline tiền xử lý video để trích xuất và phân tích nội dung đa phương tiện, bao gồm phát hiện cảnh, trích xuất keyframe, nhận dạng người dẫn tin, phân đoạn video, ASR, OCR, và mô tả hình ảnh.

## Cấu trúc thư mục dữ liệu

```
data/
├── videos/              # Video gốc đầu vào
│   ├── L01/
│   │   ├── L01_V001.mp4
│   │   ├── L01_V002.mp4
│   │   └── ...
│   ├── L02/
│   └── ...
├── shots/               # Kết quả phát hiện đoạn cắt
│   ├── L01/
│   │   ├── L01_V001_shots.json
│   │   └── ...
│   └── ...
├── keyframes/           # Khung hình trích xuất
│   ├── L01/
│   │   ├── V001/
│   │   │   ├── L01_V001_000001.jpg
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── news_anchor/         # Kết quả phát hiện người dẫn tin
│   ├── L01/
│   │   ├── L01_V001_news_anchor.json
│   │   └── ...
│   └── ...
├── news_segments/       # Kết quả phân đoạn tin tức
│   ├── L01/
│   │   ├── L01_V001_news_segment.json
│   │   └── ...
│   └── ...
├── subvideos/           # Video phụ đã cắt
│   ├── L01/
│   │   ├── V001/
│   │   │   ├── segment_001.mp4
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── transcripts/         # Kết quả ASR
│   ├── L01/
│   │   ├── segment_001_transcript.txt
│   │   └── ...
│   └── ...
├── captions/            # Kết quả mô tả hình ảnh
│   ├── L01/
│   │   ├── L01_V001_caption.json
│   │   └── ...
│   └── ...
├── ocr/                 # Kết quả OCR
│   ├── L01/
│   │   ├── L01_V001_ocr.json
│   │   └── ...
│   └── ...
├── detections/          # Kết quả phát hiện đối tượng
│   ├── L01/
│   │   ├── L01_V001_detection.json
│   │   └── ...
│   └── ...
├── embeddings/         # FAISS vector index
│   └── OpenCLIP_ViT-B-16_dfn2b_embeddings.bin
└── id2path.json         # File ánh xạ ID và đường dẫn
```

## Bảng tổng hợp các task

| Task | Input | Output | Nhiệm vụ |
|------|-------|--------|----------|
| **shot_boundary_detection** | Video files (.mp4) | JSON files chứa thông tin đoạn cắt | Phát hiện các đoạn cắt trong video để chia video thành các scene |
| **keyframe_extraction** | Video files + Shot JSON files | Image files (.jpg) | Trích xuất các khung hình đại diện từ mỗi đoạn cắt |
| **news_anchor_detection** | Keyframe images | JSON files với kết quả phân loại | Phát hiện và phân loại khung hình có người dẫn tin |
| **news_segmentation** | Keyframes + News anchor JSON | JSON files với thông tin phân đoạn | Phân đoạn video dựa trên sự xuất hiện của người dẫn tin |
| **subvideo_extraction** | Video files + News segment JSON | Subvideo files (.mp4) | Cắt video thành các đoạn phụ dựa trên kết quả phân đoạn |
| **asr** | Subvideo files | Text files (.txt) | Chuyển đổi âm thanh trong video thành văn bản |
| **remove_noise_keyframe** | Keyframes + News anchor JSON | Filtered keyframes | Loại bỏ các keyframe nhiễu không chứa thông tin quan trọng |
| **image_captioning** | Keyframe images | JSON files với mô tả | Tạo mô tả văn bản cho các keyframe |
| **ocr** | Keyframe images | JSON files với văn bản trích xuất | Trích xuất văn bản từ hình ảnh trong keyframe |
| **object_detection** | Keyframes + Caption JSON | JSON files với đối tượng phát hiện | Phát hiện và định vị các đối tượng trong keyframe |
| **save_detection_elasticsearch** | Detection JSON files | Elasticsearch index | Lưu trữ kết quả phát hiện đối tượng vào Elasticsearch |
| **save_ocr_elasticsearch** | OCR JSON files | Elasticsearch index | Lưu trữ kết quả OCR vào Elasticsearch |
| **save_embedding_faiss** | Keyframe images | FAISS index files | Tạo và lưu trữ vector embedding của keyframe vào FAISS |
| **save_caption_qdrant** | Caption JSON + Keyframes | Qdrant database | Lưu trữ embedding của caption vào Qdrant vector database |
| **build_mapping_json** | Keyframe directory | mapping.json file | Tạo file ánh xạ giữa ID và đường dẫn keyframe |

## Hướng dẫn chạy từng task

### Cài đặt

```bash
git clone <repository-url>
cd HCMAI2025
pip install -r requirements.txt
```

### 1. Phát hiện đoạn cắt (Shot Boundary Detection)

```bash
# Tất cả lessons
python preprocess.py shot_boundary_detection all data/videos data/shots

# Một lesson cụ thể
python preprocess.py shot_boundary_detection lesson data/videos data/shots --lesson_name L01
```

**Yêu cầu**: GPU (TransNetV2), môi trường Kaggle khuyến nghị

### 2. Trích xuất keyframe (Keyframe Extraction)

```bash
# Tất cả lessons
python preprocess.py keyframe_extraction all data/videos data/shots data/keyframes

# Một lesson cụ thể
python preprocess.py keyframe_extraction lesson data/videos data/shots data/keyframes --lesson_name L01
```

**Yêu cầu**: Môi trường local, không cần GPU

### 3. Phát hiện người dẫn tin (News Anchor Detection)

```bash
# Tất cả lessons
python preprocess.py news_anchor_detection all data/keyframes data/news_anchor

# Một lesson cụ thể
python preprocess.py news_anchor_detection lesson data/keyframes data/news_anchor --lesson_name L01
```

**Yêu cầu**: GPU (InternVL3), môi trường Kaggle khuyến nghị

### 4. Phân đoạn tin tức (News Segmentation)

```bash
# Tất cả lessons
python preprocess.py news_segmentation all data/keyframes data/news_anchor data/news_segments

# Một lesson cụ thể
python preprocess.py news_segmentation lesson data/keyframes data/news_anchor data/news_segments --lesson_name L01
```

**Yêu cầu**: Môi trường local, không cần GPU

### 5. Trích xuất video phụ (Subvideo Extraction)

```bash
# Tất cả lessons
python preprocess.py subvideo_extraction all data/videos data/news_segments data/subvideos /path/to/ffmpeg

# Một lesson cụ thể
python preprocess.py subvideo_extraction lesson data/videos data/news_segments data/subvideos /path/to/ffmpeg --lesson_name L01
```

**Yêu cầu**: FFmpeg, môi trường local

### 6. Nhận dạng giọng nói (ASR)

```bash
# Tất cả lessons
python preprocess.py asr all data/subvideos data/transcripts

# Một lesson cụ thể
python preprocess.py asr lesson data/subvideos data/transcripts --lesson_name L01
```

**Yêu cầu**: GPU (WhisperX), môi trường Google Colab khuyến nghị

### 7. Lọc keyframe nhiễu (Remove Noise Keyframe)

```bash
# Tất cả lessons
python preprocess.py remove_noise_keyframe all data/keyframes data/news_anchor

# Một lesson cụ thể
python preprocess.py remove_noise_keyframe lesson data/keyframes data/news_anchor --lesson_name L01
```

**Yêu cầu**: Môi trường local, không cần GPU

### 8. Mô tả hình ảnh (Image Captioning)

```bash
# Tất cả lessons
python preprocess.py image_captioning all data/keyframes data/captions

# Một lesson cụ thể
python preprocess.py image_captioning lesson data/keyframes data/captions --lesson_name L01

# Một video cụ thể
python preprocess.py image_captioning single data/keyframes data/captions --lesson_name L01 --video_name V001
```

**Yêu cầu**: GPU (InternVL3), môi trường Kaggle khuyến nghị

### 9. Nhận dạng văn bản (OCR)

```bash
# Tất cả lessons
python preprocess.py ocr all data/keyframes data/ocr

# Một lesson cụ thể
python preprocess.py ocr lesson data/keyframes data/ocr --lesson_name L01
```

**Yêu cầu**: GPU (PaddleOCR), môi trường Kaggle khuyến nghị

### 10. Phát hiện đối tượng (Object Detection)

```bash
# Tất cả lessons
python preprocess.py object_detection all data/keyframes data/captions data/detections

# Một lesson cụ thể
python preprocess.py object_detection lesson data/keyframes data/captions data/detections --lesson_name L01

# Một video cụ thể
python preprocess.py object_detection single data/keyframes data/captions data/detections --lesson_name L01 --video_name V001
```

**Yêu cầu**: GPU (GroundingDINO), môi trường Kaggle khuyến nghị

### 11. Lưu detection vào Elasticsearch

```bash
python preprocess.py save_detection_elasticsearch data/detections --index groundingdino
```

**Yêu cầu**: Elasticsearch server running, môi trường local

### 12. Lưu OCR vào Elasticsearch

```bash
python preprocess.py save_ocr_elasticsearch data/ocr --index ocr
```

**Yêu cầu**: Elasticsearch server running, môi trường local

### 13. Lưu embedding vào FAISS

```bash
python preprocess.py save_embedding_faiss data/keyframes data/faiss_index --backbone ViT-B-16 --pretrained dfn2b
```

**Yêu cầu**: Môi trường local hoặc Kaggle

### 14. Lưu caption vào Qdrant

```bash
python preprocess.py save_caption_qdrant data/captions data/keyframes data/qdrant_data --collection_name captions
```

**Yêu cầu**: Qdrant server running, môi trường local

### 15. Tạo file mapping

```bash
python preprocess.py build_mapping_json data/keyframes data/
```

**Yêu cầu**: Môi trường bất kỳ
