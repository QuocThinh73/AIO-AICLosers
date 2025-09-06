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

# CRITICAL: Set these BEFORE any paddle import to avoid MKL conflicts
os.environ['PADDLE_DISABLE_MKL'] = '1'
os.environ['PADDLE_DISABLE_MKLML'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Import utility functions
from .utils import delete_banner_and_logo


def install_paddleocr():
    """Install PaddleOCR with all required dependencies for Kaggle environment"""
    try:
        # Install required dependencies first
        print("Installing required dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "astor", "--quiet"
        ])
        
        # Install PaddlePaddle with dependencies (remove --no-deps)
        print("Installing PaddlePaddle 2.6.2...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "paddlepaddle==2.6.2", "--quiet"
        ])
        
        # Install PaddleOCR
        print("Installing PaddleOCR...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "paddleocr", "--quiet"
        ])
        
        print("PaddlePaddle 2.6.2 and PaddleOCR installed successfully with all dependencies")
    except subprocess.CalledProcessError as e:
        print(f"Primary installation failed: {e}")
        # Fallback: install latest versions
        try:
            print("Trying fallback installation with latest versions...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "astor", "paddlepaddle", "paddleocr", "--quiet"
            ])
            print("Fallback installation successful")
        except subprocess.CalledProcessError as e2:
            print(f"All installation attempts failed: {e2}")
            raise e2


def get_ocr_model():
    # Additional runtime fix for MKL issues
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="paddle")
    
    # Try to import PaddleOCR, install if not available
    try:
        # Import in isolated way to avoid conflicts
        import paddle
        paddle.disable_signal_handler()
        from paddleocr import PaddleOCR
    except ImportError:
        print("PaddleOCR not found, installing...")
        install_paddleocr()
        import paddle
        paddle.disable_signal_handler()
        from paddleocr import PaddleOCR
    except Exception as e:
        print(f"Import error: {e}, trying to reinstall...")
        install_paddleocr()
        import paddle
        paddle.disable_signal_handler()
        from paddleocr import PaddleOCR
    
    # Initialize with minimal compatible parameters only
    try:
        # Try with only supported basic parameters
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        print("PaddleOCR initialized successfully with basic configuration.")
        return ocr
    except Exception as e:
        print(f"Basic initialization failed: {e}")
        try:
            # Ultimate fallback: only language parameter
            ocr = PaddleOCR(lang='en')
            print("PaddleOCR initialized with minimal configuration.")
            return ocr
        except Exception as e2:
            print(f"Minimal initialization also failed: {e2}")
            try:
                # Last resort: no parameters at all
                ocr = PaddleOCR()
                print("PaddleOCR initialized with default configuration.")
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
