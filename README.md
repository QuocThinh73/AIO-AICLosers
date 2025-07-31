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
│ Phát hiện đối tượng│    │ OCR               │
└────────┬──────────┘    └────────┬──────────┘
         │                        │
         ↓                        ↓
┌───────────────────┐    ┌───────────────────┐
│ Chú thích hình ảnh│    │ Lưu OCR vào ES   │
└────────┬──────────┘    └────────┬──────────┘
         │                        │
         ↓                        │
┌───────────────────┐             │
│ Lưu phát hiện vào │             │
│ Elasticsearch     │             │
└────────┬──────────┘             │
         │                        │
         ↓                        ↓
┌───────────────────────────────────────────┐
│ Xây dựng tệp ánh xạ (build_mapping_json) │
└───────────────────────────────────────────┘
```

## Phụ thuộc theo giai đoạn xử lý

| Giai đoạn | Đầu vào | Đầu ra | Phụ thuộc |
|-----------|---------|--------|----------|
| shot_boundary_detection | Video gốc | File shot JSON | TransNetV2 (GPU) |
| keyframe_extraction | File shot JSON | Khung hình | Phát hiện đoạn cắt |
| news_anchor_detection | Khung hình | File phân loại JSON | InternVL3 (GPU) |
| news_segmentation | File phân loại người dẫn | File phân đoạn JSON | Phát hiện người dẫn |
| extract_subvideo | File phân đoạn tin tức | Video phụ | Phân đoạn tin tức, FFmpeg |
| asr | Video phụ | File bản ghi | WhisperX (GPU) |
| remove_noise_keyframe | Khung hình | Khung hình đã lọc | Trích xuất khung hình |
| object_detection | Khung hình | File phát hiện JSON | GroundingDINO (GPU) |
| ocr | Khung hình | File OCR JSON | EasyOCR (GPU) |
| image_captioning | Khung hình | File chú thích JSON | InternVL3 (GPU) |
| save_detection_elasticsearch | File phát hiện | ES Index | Elasticsearch, phát hiện đối tượng |
| save_ocr_elasticsearch | File OCR | ES Index | Elasticsearch, OCR |
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
# Chạy cho tất cả các bài học
python preprocess.py shot_boundary_detection --mode all --input_video_dir /path/to/videos --output_dir /path/to/output

# Chạy cho một bài học cụ thể
python preprocess.py shot_boundary_detection --mode lesson --input_video_dir /path/to/videos --output_dir /path/to/output --lesson_name L01

# Chạy cho một video cụ thể
python preprocess.py shot_boundary_detection --mode video --input_video_dir /path/to/videos --output_dir /path/to/output --lesson_name L01 --video_name V001
```

### 2. Trích xuất khung hình (Keyframe Extraction)

**Môi trường**: Local

```bash
# Chạy cho tất cả các bài học
python preprocess.py keyframe_extraction --mode all --input_shot_dir /path/to/shots --output_dir /path/to/keyframes

# Chạy cho một bài học cụ thể
python preprocess.py keyframe_extraction --mode lesson --input_shot_dir /path/to/shots --output_dir /path/to/keyframes --lesson_name L01

# Chạy cho một video cụ thể
python preprocess.py keyframe_extraction --mode video --input_shot_dir /path/to/shots --output_dir /path/to/keyframes --lesson_name L01 --video_name V001
```

### 3. Phát hiện người dẫn tin (News Anchor Detection)

**Môi trường**: Kaggle (cần GPU)

```bash
# Chạy cho tất cả các bài học
python preprocess.py news_anchor_detection --mode all --input_keyframe_dir /path/to/keyframes --output_dir /path/to/news_anchor

# Chạy cho một bài học cụ thể
python preprocess.py news_anchor_detection --mode lesson --input_keyframe_dir /path/to/keyframes --output_dir /path/to/news_anchor --lesson_name L01
```

### 4. Phân đoạn tin tức (News Segmentation)

**Môi trường**: Local

```bash
# Chạy cho tất cả các bài học
python preprocess.py segment_news --mode all --input_news_anchor_dir /path/to/news_anchor --output_dir /path/to/news_segments

# Chạy cho một bài học cụ thể
python preprocess.py segment_news --mode lesson --input_news_anchor_dir /path/to/news_anchor --output_dir /path/to/news_segments --lesson_name L01
```

### 5. Trích xuất video phụ (Extract Subvideo)

**Môi trường**: Local (cần FFmpeg)

```bash
# Chạy cho tất cả các bài học
python preprocess.py extract_subvideo --mode all --input_video_dir /path/to/videos --input_segment_dir /path/to/news_segments --output_dir /path/to/subvideos --ffmpeg_bin /path/to/ffmpeg

# Chạy cho một bài học cụ thể
python preprocess.py extract_subvideo --mode lesson --input_video_dir /path/to/videos --input_segment_dir /path/to/news_segments --output_dir /path/to/subvideos --ffmpeg_bin /path/to/ffmpeg --lesson_name L01
```

### 6. ASR (Automatic Speech Recognition)

**Môi trường**: Google Colab (cần GPU và cuDNN)

```bash
# Chạy cho tất cả các bài học
python preprocess.py asr --mode all --input_video_dir /path/to/subvideos --output_dir /path/to/transcripts

# Chạy cho một bài học cụ thể
python preprocess.py asr --mode lesson --input_video_dir /path/to/subvideos --output_dir /path/to/transcripts --lesson_name L01
```

### 7. Lọc keyframe (Remove Noise Keyframe)

**Môi trường**: Local

```bash
# Chạy cho tất cả các bài học
python preprocess.py remove_noise_keyframe --mode all --input_keyframe_dir /path/to/keyframes --output_dir /path/to/filtered_keyframes

# Chạy cho một bài học cụ thể
python preprocess.py remove_noise_keyframe --mode lesson --input_keyframe_dir /path/to/keyframes --output_dir /path/to/filtered_keyframes --lesson_name L01
```

### 8. Phát hiện đối tượng (Object Detection)

**Môi trường**: Kaggle (cần GPU)

```bash
# Chạy cho tất cả các bài học
python preprocess.py object_detection --mode all --input_keyframe_dir /path/to/filtered_keyframes --output_dir /path/to/detections

# Chạy cho một bài học cụ thể
python preprocess.py object_detection --mode lesson --input_keyframe_dir /path/to/filtered_keyframes --output_dir /path/to/detections --lesson_name L01

# Chạy cho một video cụ thể
python preprocess.py object_detection --mode video --input_keyframe_dir /path/to/filtered_keyframes --output_dir /path/to/detections --lesson_name L01 --video_name V001

# Chạy cho một khung hình cụ thể
python preprocess.py object_detection --mode single --input_keyframe_dir /path/to/filtered_keyframes --output_dir /path/to/detections --lesson_name L01 --video_name V001 --keyframe_name 000001.jpg
```

### 9. OCR (Optical Character Recognition)

**Môi trường**: Kaggle (cần GPU)

```bash
# Chạy cho tất cả các bài học
python preprocess.py ocr --mode all --input_keyframe_dir /path/to/filtered_keyframes --output_dir /path/to/ocr

# Chạy cho một bài học cụ thể
python preprocess.py ocr --mode lesson --input_keyframe_dir /path/to/filtered_keyframes --output_dir /path/to/ocr --lesson_name L01
```

### 10. Chú thích hình ảnh (Image Captioning)

**Môi trường**: Kaggle (cần GPU)

```bash
# Chạy cho tất cả các bài học
python preprocess.py image_captioning --mode all --input_keyframe_dir /path/to/filtered_keyframes --output_dir /path/to/captions

# Chạy cho một bài học cụ thể
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

### 13. Xây dựng tệp ánh xạ (Build Mapping JSON)

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
| build_mapping_json | **Local/Kaggle/Colab** | Không có yêu cầu đặc biệt |

## Lưu ý quan trọng

1. **GPU và CUDA**: Các tác vụ như phát hiện đoạn cắt, phát hiện người dẫn, object detection, OCR và image captioning đều yêu cầu GPU để đạt hiệu suất tốt.

2. **FFmpeg**: Tác vụ trích xuất video phụ yêu cầu FFmpeg đã được cài đặt. Hãy cung cấp đường dẫn đến FFmpeg binary thông qua tham số `--ffmpeg_bin`.

3. **Elasticsearch**: Các tác vụ lưu dữ liệu vào Elasticsearch yêu cầu một máy chủ Elasticsearch đang chạy.

4. **Thứ tự thực hiện**: Đảm bảo tuân theo thứ tự quy trình như đã nêu ở trên, vì các bước sau thường phụ thuộc vào đầu ra của các bước trước.

5. **ASR trên Colab**: Đối với ASR, Google Colab là lựa chọn tốt nhất vì nó đã được cài đặt sẵn cuDNN mà WhisperX cần.

6. **I/O Bound vs. Compute Bound**: Các tác vụ như news_segmentation và extract_subvideo chủ yếu là I/O bound và không cần GPU, nên chạy cục bộ hiệu quả hơn.

## Các vấn đề đã biết

1. **Kaggle và đường dẫn chỉ đọc**: Kaggle không cho phép ghi vào thư mục `/kaggle/input`. Đảm bảo đặt đường dẫn đầu ra trong `/kaggle/working`.

2. **Cài đặt thư viện trên Kaggle**: Các module đã được thiết kế để tự động cài đặt các phụ thuộc cần thiết, nhưng một số (như GroundingDINO) có thể yêu cầu thêm bước cài đặt thủ công.

3. **Lỗi WhisperX/cuDNN**: Nếu gặp lỗi cuDNN khi chạy ASR, hãy thử chạy trên Google Colab hoặc đảm bảo cuDNN được cài đặt đúng cách nếu chạy cục bộ.
