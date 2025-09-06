import os
import sys
import json
import glob
import shutil
import argparse
import cv2
import numpy as np
import subprocess
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional

# Import utility functions
from .utils import delete_banner_and_logo


def install_paddleocr():
    """Install PaddleOCR and its dependencies via subprocess for Kaggle environment"""
    try:
        # Install PaddleOCR
        subprocess.check_call([sys.executable, "-m", "pip", "install", "paddlepaddle", "--quiet"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "paddleocr", "--quiet"])
        print("PaddleOCR installed successfully via subprocess")
    except subprocess.CalledProcessError as e:
        print(f"Error installing PaddleOCR: {e}")
        raise


def get_ocr_model():
    # Try to import PaddleOCR, install if not available
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        print("PaddleOCR not found, installing...")
        install_paddleocr()
        from paddleocr import PaddleOCR
    
    # Initialize PaddleOCR without use_gpu parameter (not supported in newer versions)
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    print("PaddleOCR initialized successfully.")
    return ocr

def process_video(video_dir, output_lesson_dir, lesson_name, video_name, ocr, target_size, mask_boxes):
    keyframe_paths = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
    if keyframe_paths:
        ocr_results = process_keyframes(ocr, keyframe_paths, target_size, mask_boxes)
        
        output_file = os.path.join(output_lesson_dir, f"{lesson_name}_{video_name}_ocr.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ocr_results, f, indent=2, ensure_ascii=False)
        
        print(f"OCR results saved for {lesson_name}/{video_name}: {len(ocr_results)} keyframes processed")

def extract_text(input_dir, output_dir, mode, lesson_name=None):
    os.makedirs(output_dir, exist_ok=True)
    
    ocr = get_ocr_model()
    
    target_size = (1280, 720)
    logo_box = (1000, 50, 1300, 130)
    banner_box = (0, 660, 1280, 690)
    mask_boxes = [logo_box, banner_box]
    
    if mode == "lesson":
        lesson_dir = os.path.join(input_dir, lesson_name)
        output_lesson_dir = os.path.join(output_dir, lesson_name)
        os.makedirs(output_lesson_dir, exist_ok=True)
        for video_folder in sorted(os.listdir(lesson_dir)):
            video_dir = os.path.join(lesson_dir, video_folder)
            if os.path.isdir(video_dir):
                process_video(video_dir, output_lesson_dir, lesson_name, video_folder, ocr, target_size, mask_boxes)
    else:
        for lesson_folder in sorted(os.listdir(input_dir)):
            lesson_dir = os.path.join(input_dir, lesson_folder)
            if os.path.isdir(lesson_dir):
                output_lesson_dir = os.path.join(output_dir, lesson_folder)
                os.makedirs(output_lesson_dir, exist_ok=True)
                for video_folder in sorted(os.listdir(lesson_dir)):
                    video_dir = os.path.join(lesson_dir, video_folder)
                    if os.path.isdir(video_dir):
                        process_video(video_dir, output_lesson_dir, lesson_folder, video_folder, ocr, target_size, mask_boxes)

def process_keyframes(ocr, keyframe_paths, target_size, mask_boxes):
    ocr_results = []
    
    for keyframe_path in keyframe_paths:
        img_name = os.path.basename(keyframe_path)
        
        img = cv2.imread(keyframe_path)
        if img is None:
            continue
            
        img_resized = cv2.resize(img, target_size)
        masked_img = delete_banner_and_logo(img_resized.copy(), mask_boxes)
        masked_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
        
        # Remove cls=True parameter as it's not supported in newer PaddleOCR versions
        result = ocr.ocr(masked_rgb)
        
        frame_result = {
            "image": img_name,
            "results": []
        }
        
        if result and isinstance(result, list) and len(result) > 0:
            for line in result[0]:
                frame_result["results"].append({
                    "text": line[1][0],
                    "confidence": float(line[1][1]),
                    "box": [[float(p) for p in point] for point in line[0]]
                })
        
        ocr_results.append(frame_result)
            
    return ocr_results
