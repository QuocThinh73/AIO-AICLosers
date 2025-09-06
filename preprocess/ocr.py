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

# GPU Configuration for Kaggle EasyOCR
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU for EasyOCR

# Import utility functions
from .utils import delete_banner_and_logo


def install_easyocr():
    """Install EasyOCR with GPU support for Kaggle"""
    try:
        print("Installing EasyOCR with GPU support for Kaggle...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "easyocr", "torch", "torchvision", "--quiet"
        ])
        print("EasyOCR installed successfully with GPU support")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing EasyOCR: {e}")
        try:
            print("Trying basic EasyOCR installation...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "easyocr", "--quiet"
            ])
            return True
        except subprocess.CalledProcessError as e2:
            print(f"All EasyOCR installation attempts failed: {e2}")
            return False


def get_ocr_model():
    """Get EasyOCR model with GPU support for Kaggle"""
    try:
        import torch
        import easyocr
        
        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"GPU available: {gpu_available}")
            print(f"CUDA devices: {device_count}")
            print(f"Device name: {device_name}")
        else:
            print("GPU not available, will use CPU")
            
    except ImportError:
        print("EasyOCR not found, installing...")
        install_success = install_easyocr()
        if not install_success:
            raise RuntimeError("Failed to install EasyOCR")
        import torch
        import easyocr
        gpu_available = torch.cuda.is_available()
        
    except Exception as e:
        print(f"Import error: {e}, trying to reinstall...")
        install_easyocr()
        import torch
        import easyocr
        gpu_available = torch.cuda.is_available()
    
    try:
        print("Initializing EasyOCR...")
        # EasyOCR automatically uses GPU if available
        reader = easyocr.Reader(['en'], gpu=gpu_available)
        
        if gpu_available:
            print("EasyOCR initialized successfully with GPU acceleration.")
        else:
            print("EasyOCR initialized successfully with CPU.")
        return reader
        
    except Exception as e:
        print(f"Error initializing EasyOCR with GPU={gpu_available}: {e}")
        try:
            # Fallback to CPU
            print("Trying CPU fallback...")
            reader = easyocr.Reader(['en'], gpu=False)
            print("EasyOCR initialized with CPU fallback.")
            return reader
        except Exception as e2:
            print(f"All EasyOCR initialization attempts failed: {e2}")
            raise RuntimeError(f"Cannot initialize EasyOCR: {e2}")

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
    """Process keyframes with EasyOCR - GPU accelerated OCR processing"""
    ocr_results = []
    
    for keyframe_path in tqdm(keyframe_paths, desc="Processing keyframes with EasyOCR"):
        img_name = os.path.basename(keyframe_path)
        
        # Read image
        img = cv2.imread(keyframe_path)
        if img is None:
            continue
            
        # Use existing preprocessing from utils
        img_resized = cv2.resize(img, target_size)
        masked_img = delete_banner_and_logo(img_resized.copy(), mask_boxes)
        
        # Convert to RGB for EasyOCR
        masked_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
        
        # EasyOCR processing
        result = ocr.readtext(masked_rgb)
        
        # Format results to match expected structure
        frame_result = {
            "image": img_name,
            "results": []
        }
        
        if result:
            for detection in result:
                # EasyOCR returns (bbox, text, confidence)
                bbox, text, confidence = detection
                
                # Convert bbox format to match original structure
                # EasyOCR returns [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                frame_result["results"].append({
                    "text": text,
                    "confidence": float(confidence),
                    "box": [[float(point[0]), float(point[1])] for point in bbox]
                })
        
        ocr_results.append(frame_result)
            
    return ocr_results
