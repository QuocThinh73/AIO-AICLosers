import os
import json
import glob
import logging
from tqdm import tqdm
from typing import Dict, List, Any, Optional

# Import InternVL3 from models
from models.internvl3 import InternVL3

logger = logging.getLogger(__name__)


def process_video_keyframes(
    video_dir: str,
    captioner: InternVL3,
    output_file: str = None
) -> Dict[str, Any]:
    """Process all keyframes for a single video"""
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    video_name = os.path.basename(video_dir)
    
    # Get all keyframes for this video
    keyframes = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
    
    if not keyframes:
        logger.warning(f"No keyframes found in {video_dir}")
        return {"total": 0, "items": []}
    
    logger.info(f"Processing {len(keyframes)} keyframes for {video_name}...")
    
    # Process each keyframe
    results = []
    for keyframe_path in tqdm(keyframes, desc=f"Captioning {video_name}"):
        keyframe_name = os.path.basename(keyframe_path)
        caption = captioner.process_keyframe(keyframe_path)
        results.append({"keyframe": keyframe_name, "caption": caption})
    
    # Save results to JSON file if output_file is provided
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved caption results to {output_file}")
    
    return {"total": len(results), "items": results}


def process_lesson(
    lesson_dir: str,
    output_dir: str,
    captioner: InternVL3,
    lesson_name: str
) -> Dict[str, Any]:
    """Process all videos in a lesson"""
    # Create lesson output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all video directories in this lesson
    video_dirs = sorted(glob.glob(os.path.join(lesson_dir, "V*")))
    
    if not video_dirs:
        logger.warning(f"No video directories found in {lesson_dir}")
        return {"total": 0, "lessons": {}}
    
    logger.info(f"Processing {len(video_dirs)} videos for lesson {lesson_name}...")
    
    # Process each video
    lesson_results = {}
    for video_dir in video_dirs:
        video_name = os.path.basename(video_dir)
        output_file = os.path.join(output_dir, f"{lesson_name}_{video_name}_caption.json")
        video_result = process_video_keyframes(video_dir, captioner, output_file)
        lesson_results[video_name] = video_result
    
    return {"total": len(lesson_results), "lessons": lesson_results}


def generate_captions(
    input_keyframe_dir: str,
    output_caption_dir: str,
    mode: str,
    lesson_name: Optional[str] = None,
    video_name: Optional[str] = None
) -> Dict[str, Any]:
    """Generate captions for keyframes in specified mode"""
    os.makedirs(output_caption_dir, exist_ok=True)
    
    # Initialize model
    logger.info("Initializing InternVL3 model for image captioning...")
    captioner = InternVL3(task="image_captioning", use_quantization=True)
    
    # Handle different modes
    if mode == "all":
        # Process all lessons
        logger.info(f"Processing all lessons in {input_keyframe_dir}...")
        lesson_dirs = sorted(glob.glob(os.path.join(input_keyframe_dir, "L*")))
        
        if not lesson_dirs:
            logger.warning(f"No lesson directories found in {input_keyframe_dir}")
            return {"total": 0, "results": {}}
        
        results = {}
        for lesson_dir in lesson_dirs:
            lesson_name = os.path.basename(lesson_dir)
            lesson_result = process_lesson(lesson_dir, output_caption_dir, captioner, lesson_name)
            results[lesson_name] = lesson_result
        
        return {"total": len(results), "results": results}
    
    elif mode == "lesson":
        # Process a single lesson
        if not lesson_name:
            raise ValueError("Lesson name is required when mode is lesson")
        
        lesson_dir = os.path.join(input_keyframe_dir, lesson_name)
        if not os.path.exists(lesson_dir):
            raise ValueError(f"Lesson directory not found: {lesson_dir}")
        
        return process_lesson(lesson_dir, output_caption_dir, captioner, lesson_name)
    
    elif mode == "single":
        # Process a single video
        if not lesson_name or not video_name:
            raise ValueError("Lesson name and video name are required when mode is single")
        
        video_dir = os.path.join(input_keyframe_dir, lesson_name, video_name)
        if not os.path.exists(video_dir):
            raise ValueError(f"Video directory not found: {video_dir}")
        
        output_file = os.path.join(output_caption_dir, f"{lesson_name}_{video_name}_caption.json")
        return process_video_keyframes(video_dir, captioner, output_file)
    
    else:
        raise ValueError(f"Invalid mode: {mode}")


def caption_image(image_path="path_to_your_image.jpg"):
    """Generate caption for a single image file"""
    # Initialize model
    logger.info("Initializing InternVL3 model for image captioning...")
    captioner = InternVL3(task="image_captioning", use_quantization=True)

    # Generate caption
    caption = captioner.process_keyframe(image_path)
    print(f"Caption: {caption}")
    return caption


if __name__ == "__main__":
    # Sample usage
    import sys
    if len(sys.argv) > 1:
        caption_image(sys.argv[1])
    else:
        caption_image()