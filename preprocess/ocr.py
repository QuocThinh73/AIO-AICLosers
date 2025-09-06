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

# GPU Configuration for Kaggle
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
# Only set MKL variables if CPU fallback is needed
# os.environ['PADDLE_DISABLE_MKL'] = '1'  # Commented out for GPU usage

# Import utility functions
from .utils import delete_banner_and_logo


def install_paddleocr():
    """Install GPU-enabled PaddlePaddle and PaddleOCR for Kaggle"""
    try:
        print("Installing GPU-enabled PaddlePaddle for Kaggle...")
        # Uninstall any existing PaddlePaddle first
        subprocess.check_call([
            sys.executable, "-m", "pip", "uninstall", "paddlepaddle", "-y", "--quiet"
        ], stderr=subprocess.DEVNULL)
        
        # Try GPU-enabled PaddlePaddle first
        try:
            print("Attempting GPU-enabled PaddlePaddle installation...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "paddlepaddle-gpu==2.6.2", "-i", "https://pypi.org/simple/", "--quiet"
            ])
            print("GPU-enabled PaddlePaddle installed successfully")
        except subprocess.CalledProcessError:
            print("GPU version failed, falling back to CPU version...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "paddlepaddle==2.6.2", "-i", "https://pypi.org/simple/", "--quiet"
            ])
            print("CPU PaddlePaddle installed as fallback")
        
        # Install PaddleOCR
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "paddleocr", "--quiet"
        ])
        print("PaddleOCR installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing PaddleOCR: {e}")
        # Final fallback to basic installation
        try:
            print("Trying basic installation...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "paddlepaddle", "paddleocr", "--quiet"
            ])
            return True
        except subprocess.CalledProcessError as e2:
            print(f"All installation attempts failed: {e2}")
            return False


def get_ocr_model():
    """Get PaddleOCR model with GPU support for Kaggle"""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="paddle")
    
    try:
        # Import paddle and check GPU availability
        import paddle
        paddle.disable_signal_handler()
        from paddleocr import PaddleOCR
        
        # Check GPU availability
        gpu_available = paddle.device.cuda.device_count() > 0
        print(f"GPU available: {gpu_available}")
        if gpu_available:
            print(f"CUDA devices: {paddle.device.cuda.device_count()}")
            
    except ImportError:
        print("PaddleOCR not found, installing...")
        install_success = install_paddleocr()
        if not install_success:
            raise RuntimeError("Failed to install PaddleOCR")
        import paddle
        paddle.disable_signal_handler()
        from paddleocr import PaddleOCR
        gpu_available = paddle.device.cuda.device_count() > 0
        
    except Exception as e:
        print(f"Import error: {e}, trying to reinstall...")
        install_paddleocr()
        import paddle
        paddle.disable_signal_handler()
        from paddleocr import PaddleOCR
        gpu_available = paddle.device.cuda.device_count() > 0
    
    # Try GPU first, then fallback to CPU
    try:
        if gpu_available:
            print("Initializing PaddleOCR with GPU acceleration...")
            # For newer PaddleOCR versions, GPU is enabled by default if available
            ocr = PaddleOCR(use_angle_cls=True, lang='en', det_model_dir=None, rec_model_dir=None)
            print("PaddleOCR initialized successfully with GPU.")
        else:
            print("GPU not available, initializing with CPU...")
            ocr = PaddleOCR(use_angle_cls=True, lang='en', det_model_dir=None, rec_model_dir=None)
            print("PaddleOCR initialized successfully with CPU.")
        return ocr
        
    except Exception as e:
        print(f"Error with full parameters: {e}")
        try:
            # Fallback: basic initialization
            print("Trying fallback initialization...")
            ocr = PaddleOCR(use_angle_cls=True, lang='en')
            print("PaddleOCR initialized with basic parameters.")
            return ocr
        except Exception as e2:
            print(f"Error with basic parameters: {e2}")
            # If GPU fails, try with MKL disabled for CPU fallback
            try:
                print("GPU failed, trying CPU with MKL disabled...")
                os.environ['PADDLE_DISABLE_MKL'] = '1'
                os.environ['PADDLE_DISABLE_MKLML'] = '1'
                ocr = PaddleOCR(lang='en')
                print("PaddleOCR initialized with CPU fallback.")
                return ocr
            except Exception as e3:
                print(f"All PaddleOCR attempts failed: {e3}")
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
