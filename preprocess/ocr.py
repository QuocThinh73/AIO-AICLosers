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
    """Install PaddleOCR exactly as used in OCR.ipynb"""
    try:
        print("Installing PaddlePaddle and PaddleOCR...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "paddlepaddle", "paddleocr", "--quiet"
        ])
        print("PaddleOCR installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing PaddleOCR: {e}")
        return False


def get_ocr_model():
    """Get PaddleOCR model exactly as used in OCR.ipynb"""
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        print("PaddleOCR not found, installing...")
        install_success = install_paddleocr()
        if not install_success:
            raise RuntimeError("Failed to install PaddleOCR")
        from paddleocr import PaddleOCR
    
    try:
        # Initialize PaddleOCR with compatible parameters (use_gpu removed as it's deprecated)
        print("Initializing PaddleOCR model...")
        ocr = PaddleOCR(use_angle_cls=True, lang='en', det_model_dir=None, rec_model_dir=None)
        print("PaddleOCR initialized successfully.")
        return ocr
    except Exception as e:
        print(f"Error with full parameters: {e}")
        try:
            # Fallback: basic initialization with minimal parameters
            print("Trying fallback initialization...")
            ocr = PaddleOCR(use_angle_cls=True, lang='en')
            print("PaddleOCR initialized with basic parameters.")
            return ocr
        except Exception as e2:
            print(f"Error with basic parameters: {e2}")
            try:
                # Last resort: only language parameter
                print("Trying minimal initialization...")
                ocr = PaddleOCR(lang='en')
                print("PaddleOCR initialized with minimal parameters.")
                return ocr
            except Exception as e3:
                print(f"All initialization attempts failed: {e3}")
                raise RuntimeError(f"Cannot initialize PaddleOCR: {e3}")

def process_video(video_dir, output_lesson_dir, lesson_name, video_name, ocr, target_size, mask_boxes):
    keyframe_paths = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
    if keyframe_paths:
        ocr_results = process_keyframes(ocr, keyframe_paths, target_size, mask_boxes)
        
        output_file = os.path.join(output_lesson_dir, f"{lesson_name}_{video_name}_ocr.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ocr_results, f, ensure_ascii=False, indent=2)  # Match OCR.ipynb format
        
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
    """Process keyframes with EasyOCR - focus only on OCR functionality"""
    ocr_results = []
    
    for keyframe_path in tqdm(keyframe_paths, desc="Processing keyframes with OCR"):
        img_name = os.path.basename(keyframe_path)
        
        # Read image
        img = cv2.imread(keyframe_path)
        if img is None:
            continue
            
        # Use existing preprocessing from utils (resize + mask is handled elsewhere)
        img_resized = cv2.resize(img, target_size)
        masked_img = delete_banner_and_logo(img_resized.copy(), mask_boxes)
        
        # Convert to RGB for PaddleOCR
        masked_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
        
        # OCR processing (cls=True parameter removed as it's deprecated in newer PaddleOCR)
        result = ocr.ocr(masked_rgb)
        
        # Format results
        frame_result = {
            "image": img_name,
            "results": []
        }
        
        if result and isinstance(result, list) and len(result) > 0:
            for line in result[0]:
                # PaddleOCR returns [[bbox], [text, confidence]]
                frame_result["results"].append({
                    "text": line[1][0],
                    "confidence": float(line[1][1]),
                    "box": [[float(p) for p in point] for point in line[0]]
                })
        
        ocr_results.append(frame_result)
            
    return ocr_results
