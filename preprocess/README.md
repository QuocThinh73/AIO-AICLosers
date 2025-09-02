# HCMAI2025 - Video Preprocessing Pipeline

## Data Directory Structure
```
data/
├── videos/              # Original input videos
│   ├── L01/
│   │   ├── L01_V001.mp4
│   │   ├── L01_V002.mp4
│   │   └── ...
│   ├── L02/
│   └── ...
├── shots/               # Shot boundary detection results
│   ├── L01/
│   │   ├── L01_V001_shots.json
│   │   └── ...
│   └── ...
├── keyframes/           # Extracted keyframes
│   ├── L01/
│   │   ├── V001/
│   │   │   ├── L01_V001_000001.jpg
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── transcripts/         # ASR results
│   ├── L01/
│   │   ├── L01_V001_transcript.json
│   │   ├── L01_V002_transcript.json
│   │   └── ...
│   └── ...
├── captions/            # Image captioning results
│   ├── L01/
│   │   ├── L01_V001_caption.json
│   │   └── ...
│   └── ...
├── ocr/                 # OCR results
│   ├── L01/
│   │   ├── L01_V001_ocr.json
│   │   └── ...
│   └── ...
├── detections/          # Object detection results
│   ├── L01/
│   │   ├── L01_V001_detection.json
│   │   └── ...
│   └── ...
├── embeddings/          # FAISS vector index
│   └── OpenCLIP_ViT-B-16_dfn2b_embeddings.bin
```

## Task Summary

| Task | Input | Output | Purpose |
|------|-------|--------|---------|
| **shot_boundary_detection** | Video files (.mp4) | JSON files with shot boundaries | Detect scene changes in videos to segment into shots |
| **keyframe_extraction** | Video files + Shot JSON files | Image files (.jpg) | Extract representative frames from each shot ||
| **asr** | Subvideo files | JSON files with transcripts | Convert audio in subvideos to text transcripts |
| **image_captioning** | Keyframe images | JSON files with descriptions | Generate text descriptions for keyframes |
| **ocr** | Keyframe images | JSON files with extracted text | Extract text from images in keyframes |
| **object_detection** | Keyframes + Caption JSON | JSON files with detected objects | Detect and locate objects in keyframes |
| **save_detection_elasticsearch** | Detection JSON files | Elasticsearch index | Store object detection results in Elasticsearch |
| **save_ocr_elasticsearch** | OCR JSON files | Elasticsearch index | Store OCR results in Elasticsearch |
| **save_embedding_faiss** | Keyframe images | FAISS index files | Create and store keyframe vector embeddings in FAISS |
| **save_caption_qdrant** | Caption JSON + Keyframes | Qdrant database | Store caption embeddings in Qdrant vector database |

## Task Usage Guide

### Installation

```bash
git clone <repository-url>
cd HCMAI2025
pip install -r requirements.txt
```

### 1. Shot Boundary Detection

```bash
# All lessons
python preprocess.py shot_boundary_detection all data/videos data/shots

# Specific lesson
python preprocess.py shot_boundary_detection lesson data/videos data/shots --lesson_name L01
```

**Requirements**: GPU (TransNetV2), Kaggle environment recommended

### 2. Keyframe Extraction

```bash
# All lessons
python preprocess.py keyframe_extraction all data/videos data/shots data/keyframes

# Specific lesson
python preprocess.py keyframe_extraction lesson data/videos data/shots data/keyframes --lesson_name L01
```

**Requirements**: Local environment, no GPU needed

### 3. Automatic Speech Recognition (ASR)

```bash
# All lessons
python preprocess.py asr all data/subvideos data/transcripts

# Specific lesson
python preprocess.py asr lesson data/subvideos data/transcripts --lesson_name L01
```

**Requirements**: GPU (WhisperX), Google Colab recommended

### 4. Image Captioning

```bash
# All lessons
python preprocess.py image_captioning all data/keyframes data/captions

# Specific lesson
python preprocess.py image_captioning lesson data/keyframes data/captions --lesson_name L01

# Single video
python preprocess.py image_captioning single data/keyframes data/captions --lesson_name L01 --video_name V001
```

**Requirements**: GPU (InternVL3), Kaggle environment recommended

### 5. Optical Character Recognition (OCR)

```bash
# All lessons
python preprocess.py ocr all data/keyframes data/ocr

# Specific lesson
python preprocess.py ocr lesson data/keyframes data/ocr --lesson_name L01
```

**Requirements**: GPU (PaddleOCR), Kaggle environment recommended

### 6. Object Detection

```bash
# All lessons
python preprocess.py object_detection all data/keyframes data/captions data/detections

# Specific lesson
python preprocess.py object_detection lesson data/keyframes data/captions data/detections --lesson_name L01

# Single video
python preprocess.py object_detection single data/keyframes data/captions data/detections --lesson_name L01 --video_name V001
```

**Requirements**: GPU (GroundingDINO), Kaggle environment recommended

### 7. Save Detection to Elasticsearch

```bash
python preprocess.py save_detection_elasticsearch data/detections --index groundingdino
```

**Requirements**: Elasticsearch server running, local environment

### 8. Save OCR to Elasticsearch

```bash
python preprocess.py save_ocr_elasticsearch data/ocr --index ocr
```

**Requirements**: Elasticsearch server running, local environment

### 9. Save Embeddings to FAISS

```bash
python preprocess.py save_embedding_faiss data/keyframes data/embeddings --backbone ViT-B-16 --pretrained dfn2b
```

**Requirements**: Local or Kaggle environment

### 10. Save Captions to Qdrant

```bash
python preprocess.py save_caption_qdrant data/captions data/keyframes data/ --collection_name captions
```

**Requirements**: Qdrant server running, local environment