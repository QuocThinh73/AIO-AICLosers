import os
import sys
import json
import glob
import shutil
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional

# Import utility functions
from utils import delete_banner_and_logo


def extract_text(
    input_dir: str,
    output_dir: str,
    mode: str,
    lesson_name: str = None
) -> Dict[str, Any]:
    """
    Extract text from keyframes using OCR.
    
    Args:
        input_dir (str): Path to keyframes directory (e.g. 'database/keyframes')
        output_dir (str): Path to output directory (e.g. 'database/ocr')
        mode (str): Processing mode ('all' or 'lesson')
        lesson_name (str, optional): Name of lesson folder (e.g. 'L01'), required for 'lesson' mode
        
    Returns:
        Dict[str, Any]: Processing status and message
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize PaddleOCR
        try:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
            print("PaddleOCR initialized successfully.")
        except ImportError:
            print("PaddleOCR not installed. Installing required dependencies...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "paddlepaddle==2.6.1", "-q"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "paddleocr==2.8.1", "-q"])
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
            print("PaddleOCR installed and initialized successfully.")
        
        # Define regions to mask (banner and logo)
        target_size = (1280, 720)
        logo_box = (1000, 50, 1300, 130)
        banner_box = (0, 660, 1280, 690)
        mask_boxes = [logo_box, banner_box]
        
        if mode == "all":
            # Process all lessons
            lesson_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d)) and d.startswith("L")]
            if not lesson_dirs:
                error_msg = f"No lesson directories found in {input_dir}"
                print(f"Error: {error_msg}")
                return {"status": "error", "message": error_msg}
                
            for lesson in sorted(lesson_dirs):
                lesson_dir = os.path.join(input_dir, lesson)
                process_lesson(ocr, lesson_dir, os.path.join(output_dir, lesson), target_size, mask_boxes)
                
        elif mode == "lesson":
            # Process a specific lesson
            if not lesson_name:
                error_msg = "Lesson name is required for 'lesson' mode"
                print(f"Error: {error_msg}")
                return {"status": "error", "message": error_msg}
                
            lesson_dir = os.path.join(input_dir, lesson_name)
            if not os.path.exists(lesson_dir):
                error_msg = f"Lesson directory not found: {lesson_dir}"
                print(f"Error: {error_msg}")
                return {"status": "error", "message": error_msg}
                
            print(f"Processing lesson: {lesson_name}")
            process_lesson(ocr, lesson_dir, os.path.join(output_dir, lesson_name), target_size, mask_boxes)
            

            
        else:
            error_msg = "Invalid mode or missing required parameters"
            print(f"Error: {error_msg}")
            print("Usage: extract_text(input_dir, output_dir, mode, [lesson_name])")
            print("  - mode: 'all' or 'lesson'")
            print("  - For 'lesson' mode, provide lesson_name")
            return {"status": "error", "message": error_msg}
        
        # Zip OCR results for easy download
        zip_path = zip_ocr_results(output_dir)
        
        return {"status": "success", "message": f"OCR completed successfully. Results zipped to {zip_path}"}
        
    except Exception as e:
        print(f"Error extracting text: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


def process_lesson(ocr, lesson_dir: str, output_lesson_dir: str, target_size: Tuple[int, int], mask_boxes: List[Tuple[int, int, int, int]]) -> None:
    """
    Process all videos in a lesson directory.
    
    Args:
        ocr: PaddleOCR instance
        lesson_dir (str): Path to lesson directory
        output_lesson_dir (str): Path to output lesson directory
        target_size (tuple): Target size for image resizing
        mask_boxes (list): List of regions to mask
    """
    os.makedirs(output_lesson_dir, exist_ok=True)
    video_dirs = sorted([d for d in os.listdir(lesson_dir) if os.path.isdir(os.path.join(lesson_dir, d))])
    
    for video_dir_name in video_dirs:
        video_dir_path = os.path.join(lesson_dir, video_dir_name)
        keyframes = sorted(glob.glob(os.path.join(video_dir_path, "*.jpg")))
        
        if not keyframes:
            print(f"Warning: No keyframes found in {video_dir_path}, skipping")
            continue
            
        print(f"Processing video: {video_dir_name} ({len(keyframes)} keyframes)")
        ocr_results = process_keyframes(ocr, keyframes, target_size, mask_boxes)
        
        # Save results to JSON
        output_file = os.path.join(output_lesson_dir, f"{video_dir_name}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(ocr_results, f, ensure_ascii=False, indent=2)
            
        print(f"Results saved to: {output_file}")


def process_keyframes(ocr, keyframe_paths: List[str], target_size: Tuple[int, int], mask_boxes: List[Tuple[int, int, int, int]]) -> List[Dict[str, Any]]:
    """
    Process keyframes and extract text using OCR.
    
    Args:
        ocr: PaddleOCR instance
        keyframe_paths (list): List of keyframe paths
        target_size (tuple): Target size for image resizing
        mask_boxes (list): List of regions to mask
        
    Returns:
        list: List of OCR results for each keyframe
    """
    ocr_results = []
    
    for keyframe_path in tqdm(keyframe_paths, desc="Processing keyframes"):
        try:
            img_name = os.path.basename(keyframe_path)
            
            # Read and resize image
            img = cv2.imread(keyframe_path)
            if img is None:
                print(f"Warning: Could not read image {keyframe_path}, skipping")
                continue
                
            img_resized = cv2.resize(img, target_size)
            
            # Mask out banner and logo regions
            masked_img = delete_banner_and_logo(img_resized.copy(), mask_boxes)
            
            # Convert from BGR to RGB for OCR
            masked_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
            
            # Perform OCR
            result = ocr.ocr(masked_rgb, cls=True)
            
            # Structure OCR results
            frame_result = {
                "image": img_name,
                "results": []
            }
            
            if result and isinstance(result, list) and len(result) > 0:
                for line in result[0]:
                    frame_result["results"].append({
                        "text": line[1][0],
                        "confidence": float(line[1][1]),  # Convert numpy float to Python float for JSON serialization
                        "box": [[float(p) for p in point] for point in line[0]]  # Convert coordinates to float for JSON
                    })
            
            ocr_results.append(frame_result)
            
        except Exception as e:
            print(f"Error processing keyframe {keyframe_path}: {e}")
            continue
            
    return ocr_results


def zip_ocr_results(output_ocr_dir: str) -> str:
    """
    Create a zip archive of the OCR results for easy download.
    
    Args:
        output_ocr_dir (str): Directory containing OCR results to zip
        
    Returns:
        str: Path to the created zip file
    """
    # Get base directory and name
    base_dir = os.path.dirname(output_ocr_dir)
    dir_name = os.path.basename(output_ocr_dir)
    
    # Create zip file path
    zip_path = os.path.join(base_dir, f"{dir_name}.zip")
    
    # Print info about zip process
    print(f"Creating zip file of OCR results: {zip_path}")
    
    try:
        # Create zip file
        shutil.make_archive(
            os.path.join(base_dir, dir_name),  # Root name of the zip file
            'zip',                             # Format
            output_ocr_dir                      # Directory to zip
        )
        print(f"Zip file created successfully at: {zip_path}")
    except Exception as e:
        print(f"Warning: Failed to create zip file: {e}")
    
    return zip_path

