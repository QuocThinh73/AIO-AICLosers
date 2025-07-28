#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image Captioning Pipeline using InternVL3 model

This script provides a clean, minimal implementation for generating captions
for images using the official InternVL3 model. It handles dependency installation
and provides both batch processing and command-line interface.
"""

import os
import sys
import argparse
import glob
import json
import gc
from tqdm import tqdm

# Path to the models directory
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
from internvl3 import InternVL3


def _ensure_dependencies():
    """
    Ensure all required dependencies are installed with their latest versions.
    Special handling for Kaggle environments.
    """
    import importlib
    import subprocess
    import sys

    def _install_package(package, upgrade=False, source=None):
        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("-U")
        if source:
            cmd.append(source)
        else:
            cmd.append(package)
        subprocess.check_call(cmd)

    # Detect Kaggle environment
    is_kaggle = os.path.exists("/kaggle")
    
    # Install transformers from source on Kaggle to get latest version
    try:
        import transformers
        print(f"transformers {transformers.__version__} already installed")
        if is_kaggle:
            print("Kaggle environment detected, upgrading transformers from source...")
            _install_package("transformers", upgrade=True, source="git+https://github.com/huggingface/transformers")
    except ImportError:
        print("Installing transformers from source...")
        _install_package("transformers", source="git+https://github.com/huggingface/transformers")

    # Install or upgrade bitsandbytes for quantization
    try:
        import bitsandbytes
        print(f"bitsandbytes {bitsandbytes.__version__} already installed")
        # Always upgrade bitsandbytes on Kaggle
        if is_kaggle:
            print("Kaggle environment detected, upgrading bitsandbytes...")
            _install_package("bitsandbytes", upgrade=True)
    except ImportError:
        print("Installing bitsandbytes...")
        _install_package("bitsandbytes")

    # Install or check accelerate
    try:
        import accelerate
        print(f"accelerate {accelerate.__version__} already installed")
    except ImportError:
        print("Installing accelerate...")
        _install_package("accelerate")

    # Import torch to verify it's available
    try:
        import torch
        print(f"torch {torch.__version__} already installed")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
    except ImportError:
        print("Warning: torch not found. This may cause issues with the model.")


def generate_captions(image_paths, output_file=None):
    """
    Generate captions for images using the InternVL3 model.

    Args:
        image_paths (list): List of paths to images to caption
        output_file (str, optional): Path to output JSON file

    Returns:
        list: List of dictionaries with image paths and captions
    """
    import torch
    
    # Clear CUDA cache and run garbage collection to free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Initialize InternVL3 model with quantization
    try:
        captioner = InternVL3(task="image_captioning", use_quantization=True)
        print("Successfully loaded quantized InternVL3 model")
    except Exception as e:
        print(f"Failed to load quantized model: {e}")
        print("Falling back to non-quantized model...")
        captioner = InternVL3(task="image_captioning", use_quantization=False)
        print("Successfully loaded non-quantized InternVL3 model")
    
    # Process each image
    results = []
    for image_path in tqdm(image_paths, desc="Generating captions"):
        try:
            caption = captioner.process_keyframe(image_path)
            results.append({
                "image": os.path.basename(image_path),
                "caption": caption
            })
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            results.append({
                "image": os.path.basename(image_path),
                "caption": f"Error: {str(e)}"
            })
    
    # Save results to JSON file if specified
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results to {output_file}: {e}")
    
    return results


def process_video_directory(video_dir, output_file=None):
    """
    Process all images in a video directory.

    Args:
        video_dir (str): Path to directory containing images
        output_file (str, optional): Path to output JSON file

    Returns:
        list: List of dictionaries with image paths and captions
    """
    image_paths = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
    if not image_paths:
        print(f"No images found in {video_dir}")
        return []
    
    print(f"Found {len(image_paths)} images in {video_dir}")
    return generate_captions(image_paths, output_file)


def process_lesson_directory(lesson_dir, output_dir):
    """
    Process all videos in a lesson directory.

    Args:
        lesson_dir (str): Path to directory containing video directories
        output_dir (str): Path to directory for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    lesson_name = os.path.basename(lesson_dir)
    
    output_lesson_dir = os.path.join(output_dir, lesson_name)
    os.makedirs(output_lesson_dir, exist_ok=True)

    videos = sorted(glob.glob(os.path.join(lesson_dir, "V*")))
    if not videos:
        print(f"No video directories found in {lesson_dir}")
        return
    
    print(f"Found {len(videos)} video directories in {lesson_dir}")
    
    for video_dir in videos:
        video_name = os.path.basename(video_dir)
        output_file = os.path.join(output_lesson_dir, f"{video_name}_caption.json")
        print(f"\nProcessing video {video_name}...")
        process_video_directory(video_dir, output_file)


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description="Generate captions for images using InternVL3 model")
    parser.add_argument("input", help="Path to image file, directory of images, video directory, or lesson directory")
    parser.add_argument("-o", "--output", help="Path to output JSON file or directory")
    parser.add_argument("--batch", action="store_true", help="Process input as a lesson directory containing multiple video directories")
    
    args = parser.parse_args()
    
    # Ensure dependencies are installed
    _ensure_dependencies()
    
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' does not exist")
        return 1
    
    if args.batch:
        # Process as a lesson directory
        if args.output:
            output_dir = args.output
        else:
            output_dir = "captions"
        process_lesson_directory(args.input, output_dir)
    elif os.path.isdir(args.input):
        # Process as a video directory or directory of images
        if args.output:
            output_file = args.output
        else:
            output_file = f"{os.path.basename(args.input)}_captions.json"
        process_video_directory(args.input, output_file)
    else:
        # Process as a single image
        if args.output:
            output_file = args.output
        else:
            output_file = f"{os.path.splitext(os.path.basename(args.input))[0]}_caption.json"
        generate_captions([args.input], output_file)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())