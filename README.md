# AIO-AIClosers - Hướng dẫn tiền xử lý video

## Giới thiệu

Quy trình Preprocess.

## Quy trình xử lý (Pipeline Workflow)

```mermaid
graph TD
    A[Phát hiện đoạn cắt<br>(shot_boundary_detection)] --> B[Trích xuất khung hình<br>(keyframe_extraction)]
    B --> C[Phát hiện người dẫn<br>(news_anchor_detection)]
    C --> D[Phân đoạn tin tức<br>(news_segmentation)]
    D --> E[Trích xuất video phụ<br>(extract_subvideo)]
    E --> F1[ASR]
    E --> F2[Lọc keyframe<br>(remove_noise_keyframe)]
    F2 --> G1[Image Captioning]
    F2 --> G2[OCR]
    G1 --> H1[Object detection]
    H1 --> I1[Lưu detection vào<br>Elasticsearch]
    I1 --> Z[Xây dựng tệp ánh xạ<br>(build_mapping_json)]
    G2 --> I2[Lưu OCR vào ES]
    I2 --> Z
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
python preprocess.py shot_boundary_detection all /path/to/videos /path/to/output

# Chạy cho một Lesson cụ thể
python preprocess.py shot_boundary_detection lesson /path/to/videos /path/to/output --lesson_name L01
```

### 2. Trích xuất khung hình (Keyframe Extraction)

**Môi trường**: Local

```bash
# Chạy cho tất cả các Lesson
python preprocess.py keyframe_extraction all /path/to/videos /path/to/shots /path/to/output/keyframes

# Chạy cho một Lesson cụ thể
python preprocess.py keyframe_extraction lesson /path/to/videos /path/to/shots /path/to/output/keyframes --lesson_name L01
```

### 3. Phát hiện người dẫn tin (News Anchor Detection)

**Môi trường**: Kaggle (cần GPU)

```bash
# Chạy cho tất cả các Lesson
python preprocess.py news_anchor_detection all /path/to/keyframes /path/to/output/news_anchor

# Chạy cho một Lesson cụ thể
python preprocess.py news_anchor_detection lesson /path/to/keyframes /path/to/output/news_anchor --lesson_name L01
```

### 4. Phân đoạn tin tức (News Segmentation)

**Môi trường**: Local

```bash
# Chạy cho tất cả các Lesson
python preprocess.py segment_news all /path/to/keyframes /path/to/news_anchor /path/to/output/news_segments

# Chạy cho một Lesson cụ thể
python preprocess.py segment_news lesson /path/to/keyframes /path/to/news_anchor /path/to/output/news_segments --lesson_name L01
```

### 5. Trích xuất video phụ (Extract Subvideo)

**Môi trường**: Local (cần FFmpeg)

```bash
# Chạy cho tất cả các Lesson
python preprocess.py extract_subvideo all /path/to/videos /path/to/news_segments /path/to/output/subvideos /path/to/ffmpeg

# Chạy cho một Lesson cụ thể
python preprocess.py extract_subvideo lesson /path/to/videos /path/to/news_segments /path/to/output/subvideos /path/to/ffmpeg --lesson_name L01
```

### 6. ASR (Automatic Speech Recognition)

**Môi trường**: Google Colab (cần GPU và cuDNN)

```bash
# Chạy cho tất cả các Lesson
python preprocess.py asr all /path/to/subvideos /path/to/output/transcripts

# Chạy cho một Lesson cụ thể
python preprocess.py asr lesson /path/to/subvideos /path/to/output/transcripts --lesson_name L01
```

### 7. Lọc keyframe (Remove Noise Keyframe)

**Môi trường**: Local

```bash
# Chạy cho tất cả các Lesson
python preprocess.py remove_noise_keyframe all /path/to/keyframes /path/to/output/filtered_keyframes

# Chạy cho một Lesson cụ thể
python preprocess.py remove_noise_keyframe lesson /path/to/keyframes /path/to/output/filtered_keyframes --lesson_name L01
```

### 8. Phát hiện đối tượng (Object Detection)

**Môi trường**: Kaggle (cần GPU)

```bash
# Chạy cho tất cả các Lesson
python preprocess.py object_detection all /path/to/keyframes /path/to/captions /path/to/output/detections

# Chạy cho một Lesson cụ thể
python preprocess.py object_detection lesson /path/to/keyframes /path/to/captions /path/to/output/detections --lesson_name L01

# Chạy cho một video cụ thể
python preprocess.py object_detection video /path/to/keyframes /path/to/captions /path/to/output/detections --lesson_name L01 --video_name V001
```

### 9. OCR (Optical Character Recognition)

**Môi trường**: Kaggle (cần GPU)

```bash
# Chạy cho tất cả các Lesson
python preprocess.py ocr all /path/to/keyframes /path/to/output/ocr

# Chạy cho một Lesson cụ thể
python preprocess.py ocr lesson /path/to/keyframes /path/to/output/ocr --lesson_name L01
```

### 10. Chú thích hình ảnh (Image Captioning)

**Môi trường**: Kaggle (cần GPU)

```bash
# Chạy cho tất cả các Lesson
python preprocess.py image_captioning all /path/to/keyframes /path/to/output/captions

# Chạy cho một Lesson cụ thể
python preprocess.py image_captioning lesson /path/to/keyframes /path/to/output/captions --lesson_name L01
```

### 11. Lưu phát hiện vào Elasticsearch

**Môi trường**: Local (cần Elasticsearch)

```bash
python preprocess.py save_detection_elasticsearch /path/to/detections --index groundingdino
```

### 12. Lưu OCR vào Elasticsearch

**Môi trường**: Local (cần Elasticsearch)

```bash
python preprocess.py save_ocr_elasticsearch /path/to/ocr --index ocr
```

### 13. Lưu embedding vào Faiss

**Môi trường**: Local

```bash
python preprocess.py save_embedding_faiss /path/to/keyframes /path/to/faiss_index --backbone ViT-B-16 --pretrained dfn2b
```

### 14. Lưu caption vào Qdrant

**Môi trường**: Local

```bash
python preprocess.py save_caption_qdrant /path/to/captions /path/to/keyframes /path/to/output_dir --collection_name captions
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

6. **Mấy cái có -- phía trước**: Là mấy cái optional.
