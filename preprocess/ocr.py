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


def install_easyocr():
    """Install EasyOCR and its dependencies for Kaggle environment"""
    try:
        print("Installing EasyOCR and dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "easyocr", "--quiet"
        ])
        print("EasyOCR installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing EasyOCR: {e}")
        return False


def get_ocr_model():
    """Get EasyOCR model with English language support"""
    try:
        import easyocr
    except ImportError:
        print("EasyOCR not found, installing...")
        install_success = install_easyocr()
        if not install_success:
            raise RuntimeError("Failed to install EasyOCR")
        import easyocr
    
    try:
        # Initialize EasyOCR with English language
        print("Initializing EasyOCR model...")
        reader = easyocr.Reader(['en'], gpu=False)
        print("EasyOCR initialized successfully.")
        return reader
    except Exception as e:
        print(f"Error initializing EasyOCR: {e}")
        raise RuntimeError(f"Cannot initialize EasyOCR: {e}")

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
    """Process keyframes with EasyOCR to extract text"""
    ocr_results = []
    
    for keyframe_path in tqdm(keyframe_paths, desc="Processing keyframes with OCR"):
        img_name = os.path.basename(keyframe_path)
        
        # Read and preprocess image
        img = cv2.imread(keyframe_path)
        if img is None:
            continue
            
        img_resized = cv2.resize(img, target_size)
        masked_img = delete_banner_and_logo(img_resized.copy(), mask_boxes)
        
        # EasyOCR expects RGB format
        masked_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
        
        # Run OCR (EasyOCR format is different from PaddleOCR)
        result = ocr.readtext(masked_rgb)
        
        # Format results to match original structure
        frame_result = {
            "image": img_name,
            "results": []
        }
        
        if result:
            for detection in result:
                # EasyOCR returns (bbox, text, confidence)
                bbox, text, confidence = detection
                
                # Convert bbox to match expected format
                # EasyOCR returns [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                frame_result["results"].append({
                    "text": text,
                    "confidence": float(confidence),
                    "box": [[float(point[0]), float(point[1])] for point in bbox]
                })
        
        ocr_results.append(frame_result)
            
    return ocr_results
