# AIO-AIClosers - Hướng dẫn tiền xử lý video

## Giới thiệu

Quy trình Preprocess.

## Quy trình xử lý (Pipeline Workflow)

```
┌───────────────────────┐
│ Phát hiện đoạn cắt    │ (shot_boundary_detection)
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│ Trích xuất khung hình │ (keyframe_extraction)
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│ Phát hiện người dẫn   │ (news_anchor_detection)
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│ Phân đoạn tin tức     │ (news_segmentation)
└───────────┬───────────┘
            ↓
┌───────────────────────┐
│ Trích xuất video phụ  │ (extract_subvideo)
└───────────┬───────────┘
            ↓
   ┌────────┴────────┐
   ↓                 ↓
┌──────────┐  ┌───────────────┐
│ ASR      │  │ Lọc keyframe  │ (remove_noise_keyframe)
└──────────┘  └───────┬───────┘
                      ↓
         ┌────────────┴────────────┐
         ↓                         ↓
┌───────────────────┐    ┌───────────────────┐
│ Image Captioning  │    │        OCR        │
└────────┬──────────┘    └────────┬──────────┘
         │                        │
         ↓                        ↓
┌───────────────────┐    ┌───────────────────┐
│ Object detection  │    │  Lưu OCR vào ES   │
└────────┬──────────┘    └────────┬──────────┘
         │                        │
         ↓                        │
┌───────────────────┐             │
│ Lưu detection vào │             │
│ Elasticsearch     │             │
└────────┬──────────┘             │
         │                        │
         ↓                        ↓
┌───────────────────────────────────────────┐
│ Xây dựng tệp ánh xạ (build_mapping_json)  │
└───────────────────────────────────────────┘
```

## Phụ thuộc theo giai đoạn xử lý

| Giai đoạn | Đầu vào | Đầu ra | Phụ thuộc |
|-----------|---------|--------|----------|
| shot_boundary_detection | Video gốc | File shot JSON | TransNetV2 (GPU) |
| keyframe_extraction | File Video gốc và File shot JSON | Khung hình | Phát hiện đoạn cắt |
| news_anchor_detection | Khung hình | File phân loại JSON | InternVL3 (GPU) |
| news_segmentation | File keyframes và File news anchor JSON | File phân đoạn JSON | Phát hiện người dẫn |
| extract_subvideo | File video gốc và File phân đoạn JSON | Video phụ | Phân đoạn tin tức, FFmpeg |
| asr | File SubVideo | File bản ghi | WhisperX (GPU) |
| remove_noise_keyframe | File Keyframes và File news anchor JSON | Khung hình đã lọc | Trích xuất khung hình |
| object_detection | File Keyframes và File Caption JSON | File phát hiện JSON | GroundingDINO (GPU) |
| ocr | File Keyframes | File OCR JSON | EasyOCR (GPU) |
| image_captioning | File Keyframes | File chú thích JSON | InternVL3 (GPU) |
| save_detection_elasticsearch | File detection JSON | ES Index | Elasticsearch, phát hiện đối tượng |
| save_ocr_elasticsearch | File OCR JSON | ES Index | Elasticsearch, OCR |
| save_embedding_faiss | File Keyframes | Faiss Index | Faiss |
| save_caption_qdrant | File Caption JSON | Qdrant Index | Qdrant |
| build_mapping_json | Thư mục đầu ra | mapping.json | Các giai đoạn khác |

## Hướng dẫn chạy

### Cài đặt

```bash
# Clone repo
git clone https://github.com/yourusername/AIO-AIClosers.git
cd AIO-AIClosers
```

### 1. Phát hiện đoạn cắt (Shot Boundary Detection)

**Môi trường**: Kaggle (cần GPU)

```bash
# Chạy cho tất cả các Lesson
python preprocess.py shot_boundary_detection --mode all --input_video_dir /path/to/videos --output_dir /path/to/output

# Chạy cho một Lesson cụ thể
python preprocess.py shot_boundary_detection --mode lesson --input_video_dir /path/to/videos --output_dir /path/to/output --lesson_name L01

# Chạy cho một video cụ thể
python preprocess.py shot_boundary_detection --mode video --input_video_dir /path/to/videos --output_dir /path/to/output --lesson_name L01 --video_name V001
```

### 2. Trích xuất khung hình (Keyframe Extraction)

**Môi trường**: Local

```bash
# Chạy cho tất cả các Lesson
python preprocess.py keyframe_extraction --mode all --input_shot_dir /path/to/shots --output_dir /path/to/keyframes

# Chạy cho một Lesson cụ thể
python preprocess.py keyframe_extraction --mode lesson --input_shot_dir /path/to/shots --output_dir /path/to/keyframes --lesson_name L01

# Chạy cho một video cụ thể
python preprocess.py keyframe_extraction --mode video --input_shot_dir /path/to/shots --output_dir /path/to/keyframes --lesson_name L01 --video_name V001
```

### 3. Phát hiện người dẫn tin (News Anchor Detection)

**Môi trường**: Kaggle (cần GPU)

```bash
# Chạy cho tất cả các Lesson
python preprocess.py news_anchor_detection --mode all --input_keyframe_dir /path/to/keyframes --output_dir /path/to/news_anchor

# Chạy cho một Lesson cụ thể
python preprocess.py news_anchor_detection --mode lesson --input_keyframe_dir /path/to/keyframes --output_dir /path/to/news_anchor --lesson_name L01
```

### 4. Phân đoạn tin tức (News Segmentation)

**Môi trường**: Local

```bash
# Chạy cho tất cả các Lesson
python preprocess.py segment_news --mode all --input_news_anchor_dir /path/to/news_anchor --output_dir /path/to/news_segments

# Chạy cho một Lesson cụ thể
python preprocess.py segment_news --mode lesson --input_news_anchor_dir /path/to/news_anchor --output_dir /path/to/news_segments --lesson_name L01
```

### 5. Trích xuất video phụ (Extract Subvideo)

**Môi trường**: Local (cần FFmpeg)

```bash
# Chạy cho tất cả các Lesson
python preprocess.py extract_subvideo --mode all --input_video_dir /path/to/videos --input_segment_dir /path/to/news_segments --output_dir /path/to/subvideos --ffmpeg_bin /path/to/ffmpeg

# Chạy cho một Lesson cụ thể
python preprocess.py extract_subvideo --mode lesson --input_video_dir /path/to/videos --input_segment_dir /path/to/news_segments --output_dir /path/to/subvideos --ffmpeg_bin /path/to/ffmpeg --lesson_name L01
```

### 6. ASR (Automatic Speech Recognition)

**Môi trường**: Google Colab (cần GPU và cuDNN)

```bash
# Chạy cho tất cả các Lesson
python preprocess.py asr --mode all --input_video_dir /path/to/subvideos --output_dir /path/to/transcripts

# Chạy cho một Lesson cụ thể
python preprocess.py asr --mode lesson --input_video_dir /path/to/subvideos --output_dir /path/to/transcripts --lesson_name L01
```

### 7. Lọc keyframe (Remove Noise Keyframe)

**Môi trường**: Local

```bash
# Chạy cho tất cả các Lesson
python preprocess.py remove_noise_keyframe --mode all --input_keyframe_dir /path/to/keyframes --output_dir /path/to/filtered_keyframes

# Chạy cho một Lesson cụ thể
python preprocess.py remove_noise_keyframe --mode lesson --input_keyframe_dir /path/to/keyframes --output_dir /path/to/filtered_keyframes --lesson_name L01
```

### 8. Phát hiện đối tượng (Object Detection)

**Môi trường**: Kaggle (cần GPU)

```bash
# Chạy cho tất cả các Lesson
python preprocess.py object_detection --mode all --input_keyframe_dir /path/to/filtered_keyframes --output_dir /path/to/detections

# Chạy cho một Lesson cụ thể
python preprocess.py object_detection --mode lesson --input_keyframe_dir /path/to/filtered_keyframes --output_dir /path/to/detections --lesson_name L01

# Chạy cho một video cụ thể
python preprocess.py object_detection --mode video --input_keyframe_dir /path/to/filtered_keyframes --output_dir /path/to/detections --lesson_name L01 --video_name V001
```

### 9. OCR (Optical Character Recognition)

**Môi trường**: Kaggle (cần GPU)

```bash
# Chạy cho tất cả các Lesson
python preprocess.py ocr --mode all --input_keyframe_dir /path/to/filtered_keyframes --output_dir /path/to/ocr

# Chạy cho một Lesson cụ thể
python preprocess.py ocr --mode lesson --input_keyframe_dir /path/to/filtered_keyframes --output_dir /path/to/ocr --lesson_name L01
```

### 10. Chú thích hình ảnh (Image Captioning)

**Môi trường**: Kaggle (cần GPU)

```bash
# Chạy cho tất cả các Lesson
python preprocess.py image_captioning --mode all --input_keyframe_dir /path/to/filtered_keyframes --output_dir /path/to/captions

# Chạy cho một Lesson cụ thể
python preprocess.py image_captioning --mode lesson --input_keyframe_dir /path/to/filtered_keyframes --output_dir /path/to/captions --lesson_name L01
```

### 11. Lưu phát hiện vào Elasticsearch

**Môi trường**: Local (cần Elasticsearch)

```bash
python preprocess.py save_detection_elasticsearch --input_dir /path/to/detections --es_host localhost --es_port 9200
```

### 12. Lưu OCR vào Elasticsearch

**Môi trường**: Local (cần Elasticsearch)

```bash
python preprocess.py save_ocr_elasticsearch --input_dir /path/to/ocr --es_host localhost --es_port 9200
```

### 13. Lưu embedding vào Faiss

**Môi trường**: Local

```bash
python preprocess.py save_embedding_faiss --input_keyframe_dir /path/to/keyframes --output_dir /path/to/faiss_index
```

### 14. Lưu caption vào Qdrant

**Môi trường**: Local

```bash
python preprocess.py save_caption_qdrant --input_dir /path/to/captions --qdrant_url http://localhost:6333
```

### 15. Xây dựng tệp ánh xạ (Build Mapping JSON)

**Môi trường**: Local/Kaggle/Colab

```bash
python preprocess.py build_mapping_json --output_dir /path/to/output_dir
```

## Phân loại môi trường thực thi

| Module | Môi trường thực thi | Lý do |
|--------|---------------------|-------|
| shot_boundary_detection | **Kaggle** | Cần GPU để xử lý nhanh TransNetV2 |
| keyframe_extraction | **Local** | Không cần GPU, chỉ xử lý I/O |
| news_anchor_detection | **Kaggle** | Cần GPU để chạy mô hình InternVL3 |
| news_segmentation | **Local** | Không cần GPU, chỉ phân tích JSON |
| extract_subvideo | **Local** | Cần FFmpeg và xử lý I/O lớn |
| asr | **Google Colab** | Cần GPU và cuDNN được cài đặt sẵn |
| remove_noise_keyframe | **Local** | Không cần GPU, chỉ phân tích hình ảnh đơn giản |
| object_detection | **Kaggle** | Cần GPU để chạy GroundingDINO |
| ocr | **Kaggle** | Cần GPU để chạy EasyOCR hiệu quả |
| image_captioning | **Kaggle** | Cần GPU để chạy InternVL3 |
| save_detection_elasticsearch | **Local** | Cần kết nối Elasticsearch |
| save_ocr_elasticsearch | **Local** | Cần kết nối Elasticsearch |
| save_embedding_faiss | **Local** | Không cần GPU |
| save_caption_qdrant | **Local** | Không cần GPU |
| build_mapping_json | **Local/Kaggle/Colab** | Không có yêu cầu đặc biệt |

## Lưu ý quan trọng

1. **FFmpeg**: Tác vụ trích xuất video phụ yêu cầu FFmpeg đã được cài đặt. Hãy cung cấp đường dẫn đến FFmpeg binary thông qua tham số `--ffmpeg_bin`.

2. **Elasticsearch**: Các tác vụ lưu dữ liệu vào Elasticsearch yêu cầu một máy chủ Elasticsearch đang chạy.

3. **Thứ tự thực hiện**: Đảm bảo tuân theo thứ tự quy trình như đã nêu ở trên, vì các bước sau thường phụ thuộc vào đầu ra của các bước trước.

4. **ASR trên Colab**: Đối với ASR, Google Colab là lựa chọn tốt nhất vì nó đã được cài đặt sẵn cuDNN mà WhisperX cần.

5. **I/O Bound vs. Compute Bound**: Các tác vụ như news_segmentation và extract_subvideo chủ yếu là I/O bound và không cần GPU, nên chạy cục bộ hiệu quả hơn.
