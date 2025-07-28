#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Auto-install dependencies
import sys
import os
import subprocess
import importlib.util


def _ensure_dependencies():
    """Ensure all required dependencies are installed with latest versions."""
    # Check if running in Kaggle environment
    in_kaggle = 'google.colab' in sys.modules or 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
    
    # Required packages
    required_packages = [
        "transformers",  # For model loading
        "bitsandbytes",  # For quantization
        "accelerate",   # For optimized inference
        "torch"         # PyTorch
    ]
    
    for package in required_packages:
        # Skip torch installation as it's pre-installed in most environments with correct CUDA versions
        if package == "torch":
            continue
            
        try:
            if package == "transformers" and in_kaggle:
                # On Kaggle, always install latest transformers from source
                print(f"Installing {package} from source (required for latest InternVL3 support)...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", 
                                      "--upgrade", "git+https://github.com/huggingface/transformers"])
            else:
                # For other packages, install latest version
                print(f"Installing/upgrading {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
                
        except Exception as e:
            print(f"Warning: Failed to install/upgrade {package}. Error: {e}")
            # Continue anyway, as the package might already be installed


# Install dependencies first
_ensure_dependencies()

# Now import required modules
import os
import json
import glob
from tqdm import tqdm

# Import our model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.internvl3 import InternVL3


def generate_captions(
    input_dir: str,
    output_dir: str,
    mode: str,
    lesson_name: str = None,
    video_name: str = None
) -> dict:
    """
    Generate captions for keyframes using the InternVL3 model.
    
    Args:
        input_dir (str): Base directory for keyframes
        output_dir (str): Output directory for captions
        mode (str): Processing mode - "all", "lesson", or "single"
        lesson_name (str, optional): Name of the lesson to process (for "lesson" and "single" modes)
        video_name (str, optional): Name of the video to process (for "single" mode)
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize the model (use quantization if CUDA is available)
        model = InternVL3(task="image_captioning", use_quantization=True)
        print(f"Initialized InternVL3 model on {model.device} device")
        
        # Process based on the selected mode
        if mode == "all":
            # Process all lessons
            lessons = sorted(glob.glob(os.path.join(input_dir, "L*")))
            for lesson_dir in lessons:
                lesson_id = os.path.basename(lesson_dir)
                print(f"Processing lesson: {lesson_id}")
                model.process_batch(lesson_dir, output_dir)
                
        elif mode == "lesson" and lesson_name:
            # Process a specific lesson
            lesson_dir = os.path.join(input_dir, lesson_name)
            if not os.path.exists(lesson_dir):
                error_msg = f"Lesson directory not found: {lesson_dir}"
                print(f"Error: {error_msg}")
                return {"status": "error", "message": error_msg}
                
            print(f"Processing lesson: {lesson_name}")
            model.process_batch(lesson_dir, output_dir)
            
        elif mode == "single" and lesson_name and video_name:
            # Process a single video
            video_dir = os.path.join(input_dir, lesson_name, video_name)
            if not os.path.exists(video_dir):
                error_msg = f"Video directory not found: {video_dir}"
                print(f"Error: {error_msg}")
                return {"status": "error", "message": error_msg}
                
            # Create output directory structure
            output_lesson_dir = os.path.join(output_dir, lesson_name)
            os.makedirs(output_lesson_dir, exist_ok=True)
            
            # Process the video
            keyframes = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
            if not keyframes:
                error_msg = f"No keyframes found in {video_dir}"
                print(f"Warning: {error_msg}")
                return {"status": "error", "message": error_msg}
                
            video_results = []
            for keyframe_path in tqdm(keyframes, desc=f"Processing {lesson_name}/{video_name}"):
                keyframe_name = os.path.basename(keyframe_path)
                caption = model.process_keyframe(keyframe_path)
                video_results.append({
                    "keyframe": keyframe_name,
                    "caption": caption
                })
                
            # Save results
            output_file = os.path.join(output_lesson_dir, f"{video_name}_image_captioning.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(video_results, f, indent=2, ensure_ascii=False)
                
            print(f"Results saved to: {output_file}")
            
        else:
            error_msg = "Invalid mode or missing required parameters"
            print(f"Error: {error_msg}")
            print("Usage: generate_captions(input_dir, output_dir, mode, [lesson_name], [video_name])")
            print("  - mode: 'all', 'lesson', or 'single'")
            print("  - For 'lesson' mode, provide lesson_name")
            print("  - For 'single' mode, provide both lesson_name and video_name")
            return {"status": "error", "message": error_msg}
            
        # Return results as dictionary (similar to object_detection.py)
        return {"status": "success", "message": "Caption generation completed successfully"}
        
    except Exception as e:
        print(f"Error generating captions: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


# If the script is run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate captions for keyframes using InternVL3")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "lesson", "single"],
                        help="Processing mode: all lessons, single lesson, or single video")
    parser.add_argument("--input_dir", type=str, default="database/keyframes",
                        help="Base directory containing keyframes")
    parser.add_argument("--output_dir", type=str, default="database/caption",
                        help="Output directory for saving captions")
    parser.add_argument("--lesson", type=str, help="Lesson name (required for 'lesson' and 'single' modes)")
    parser.add_argument("--video", type=str, help="Video name (required for 'single' mode)")
    
    args = parser.parse_args()
    
    # Call the main function
    result = generate_captions(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        lesson_name=args.lesson,
        video_name=args.video
    )
    
    if result.get("status") == "error":
        print(f"Error: {result.get('message')}")
        sys.exit(1)
    else:
        print(f"Success: {result.get('message')}")
        sys.exit(0)
