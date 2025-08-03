# HCMAI2025 - Video Preprocessing Pipeline

## Overview

A comprehensive video preprocessing pipeline for extracting and analyzing multimedia content, including shot detection, keyframe extraction, news anchor detection, video segmentation, ASR, OCR, and image captioning.

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
├── news_anchor/         # News anchor detection results
│   ├── L01/
│   │   ├── L01_V001_news_anchor.json
│   │   └── ...
│   └── ...
├── news_segments/       # News segmentation results
│   ├── L01/
│   │   ├── L01_V001_news_segment.json
│   │   └── ...
│   └── ...
├── subvideos/           # Extracted subvideos
│   ├── L01/
│   │   ├── L01_V001/
│   │   │   ├── L01_V001_000260_001125.mp4
│   │   │   ├── L01_V001_001168_003635.mp4
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
└── id2path.json         # ID to path mapping file
```

## Task Summary

| Task | Input | Output | Purpose |
|------|-------|--------|---------|
| **shot_boundary_detection** | Video files (.mp4) | JSON files with shot boundaries | Detect scene changes in videos to segment into shots |
| **keyframe_extraction** | Video files + Shot JSON files | Image files (.jpg) | Extract representative frames from each shot |
| **news_anchor_detection** | Keyframe images | JSON files with classification results | Detect and classify frames containing news anchors |
| **news_segmentation** | Keyframes + News anchor JSON | JSON files with segment information | Segment videos based on news anchor appearance |
| **subvideo_extraction** | Video files + News segment JSON | Subvideo files (.mp4) | Cut videos into segments based on segmentation results |
| **asr** | Subvideo files | JSON files with transcripts | Convert audio in subvideos to text transcripts |
| **remove_noise_keyframe** | Keyframes + News anchor JSON | Filtered keyframes | Remove noisy keyframes without important information |
| **image_captioning** | Keyframe images | JSON files with descriptions | Generate text descriptions for keyframes |
| **ocr** | Keyframe images | JSON files with extracted text | Extract text from images in keyframes |
| **object_detection** | Keyframes + Caption JSON | JSON files with detected objects | Detect and locate objects in keyframes |
| **save_detection_elasticsearch** | Detection JSON files | Elasticsearch index | Store object detection results in Elasticsearch |
| **save_ocr_elasticsearch** | OCR JSON files | Elasticsearch index | Store OCR results in Elasticsearch |
| **save_embedding_faiss** | Keyframe images | FAISS index files | Create and store keyframe vector embeddings in FAISS |
| **save_caption_qdrant** | Caption JSON + Keyframes | Qdrant database | Store caption embeddings in Qdrant vector database |
| **build_mapping_json** | Keyframe directory | mapping.json file | Create mapping file between IDs and keyframe paths |

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

### 3. News Anchor Detection

```bash
# All lessons
python preprocess.py news_anchor_detection all data/keyframes data/news_anchor

# Specific lesson
python preprocess.py news_anchor_detection lesson data/keyframes data/news_anchor --lesson_name L01
```

**Requirements**: GPU (InternVL3), Kaggle environment recommended

### 4. News Segmentation

```bash
# All lessons
python preprocess.py news_segmentation all data/keyframes data/news_anchor data/news_segments

# Specific lesson
python preprocess.py news_segmentation lesson data/keyframes data/news_anchor data/news_segments --lesson_name L01
```

**Requirements**: Local environment, no GPU needed

### 5. Subvideo Extraction

```bash
# All lessons
python preprocess.py subvideo_extraction all data/videos data/news_segments data/subvideos /path/to/ffmpeg

# Specific lesson
python preprocess.py subvideo_extraction lesson data/videos data/news_segments data/subvideos /path/to/ffmpeg --lesson_name L01
```

**Requirements**: FFmpeg, local environment

### 6. Automatic Speech Recognition (ASR)

```bash
# All lessons
python preprocess.py asr all data/subvideos data/transcripts

# Specific lesson
python preprocess.py asr lesson data/subvideos data/transcripts --lesson_name L01
```

**Requirements**: GPU (WhisperX), Google Colab recommended

### 7. Remove Noise Keyframes

```bash
# All lessons
python preprocess.py remove_noise_keyframe all data/keyframes data/news_anchor

# Specific lesson
python preprocess.py remove_noise_keyframe lesson data/keyframes data/news_anchor --lesson_name L01
```

**Requirements**: Local environment, no GPU needed

### 8. Image Captioning

```bash
# All lessons
python preprocess.py image_captioning all data/keyframes data/captions

# Specific lesson
python preprocess.py image_captioning lesson data/keyframes data/captions --lesson_name L01

# Single video
python preprocess.py image_captioning single data/keyframes data/captions --lesson_name L01 --video_name V001
```

**Requirements**: GPU (InternVL3), Kaggle environment recommended

### 9. Optical Character Recognition (OCR)

```bash
# All lessons
python preprocess.py ocr all data/keyframes data/ocr

# Specific lesson
python preprocess.py ocr lesson data/keyframes data/ocr --lesson_name L01
```

**Requirements**: GPU (PaddleOCR), Kaggle environment recommended

### 10. Object Detection

```bash
# All lessons
python preprocess.py object_detection all data/keyframes data/captions data/detections

# Specific lesson
python preprocess.py object_detection lesson data/keyframes data/captions data/detections --lesson_name L01

# Single video
python preprocess.py object_detection single data/keyframes data/captions data/detections --lesson_name L01 --video_name V001
```

**Requirements**: GPU (GroundingDINO), Kaggle environment recommended

### 11. Save Detection to Elasticsearch

```bash
python preprocess.py save_detection_elasticsearch data/detections --index groundingdino
```

**Requirements**: Elasticsearch server running, local environment

### 12. Save OCR to Elasticsearch

```bash
python preprocess.py save_ocr_elasticsearch data/ocr --index ocr
```

**Requirements**: Elasticsearch server running, local environment

### 13. Save Embeddings to FAISS

```bash
python preprocess.py save_embedding_faiss data/keyframes data/embeddings --backbone ViT-B-16 --pretrained dfn2b
```

**Requirements**: Local or Kaggle environment

### 14. Save Captions to Qdrant

```bash
python preprocess.py save_caption_qdrant data/captions data/keyframes data/ --collection_name captions
```

**Requirements**: Qdrant server running, local environment

### 15. Build Mapping JSON

```bash
python preprocess.py build_mapping_json data/keyframes data/
```

**Requirements**: Any environment
