import os
import json
import glob
from typing import Dict, List, Any
from PIL import Image

from models.groundingdino import GroundingDINO

def extract_objects_from_caption(caption: str) -> List[str]:
    """
    Extract meaningful segments from captions to use as prompts for object detection
    """
    # Remove unnecessary meta text
    caption = caption.replace("The image appears to be", "")
    caption = caption.replace("The image shows", "")
    
    # List to store prompt segments
    prompts = []
    
    # Split caption into parts by line breaks
    parts = caption.split('\n')
    
    for part in parts:
        # Skip short lines or lines without content
        if len(part.strip()) < 10:
            continue
            
        # Find specific description sections
        if ':' in part and '**' in part:
            # Example: "**Top Row:** - People walking..."
            topic_parts = part.split(':')
            if len(topic_parts) > 1 and len(topic_parts[1].strip()) > 10:
                prompts.append(topic_parts[1].strip())
        elif '-' in part:
            # Split into parts by hyphens
            bullet_points = part.split('-')
            for point in bullet_points:
                if len(point.strip()) > 10:
                    prompts.append(point.strip())
        elif len(part.strip()) > 20 and part.strip().endswith('.'):
            # Get complete sentences
            sentences = part.split('.')
            for sentence in sentences:
                if len(sentence.strip()) > 20:
                    prompts.append(sentence.strip() + '.')
    
    # If no prompts were found, use the original caption
    if not prompts:
        # If caption is too long, split into smaller segments
        if len(caption) > 200:
            sentences = caption.split('.')
            for sentence in sentences:
                if len(sentence.strip()) > 20:
                    prompts.append(sentence.strip() + '.')
        else:
            prompts.append(caption)
    
    # Limit the number of prompts to avoid overload
    return prompts[:3]

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    # Box coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate area of each box
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate coordinates of intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Check if there is no intersection
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    # Calculate area of intersection
    area_intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate IoU
    iou = area_intersection / (area1 + area2 - area_intersection)
    
    return iou

def filter_objects(objects, iou_threshold=0.7, confidence_threshold=0.5):
    """Filter duplicated objects, keep only the highest scoring object for each group of overlapping boxes"""
    # If there are no objects, return empty list
    if not objects:
        return []
    
    # Filter objects based on confidence threshold
    objects = [obj for obj in objects if obj["score"] >= confidence_threshold]
    
    # Sort objects by score in descending order
    sorted_objects = sorted(objects, key=lambda x: x["score"], reverse=True)
    
    # List to store filtered objects
    filtered_objects = []
    
    # Iterate through each object
    for obj in sorted_objects:
        # Check if current object overlaps with any object in filtered_objects
        duplicate = False
        for filtered_obj in filtered_objects:
            # If same object name and IoU greater than threshold
            if obj["object"] == filtered_obj["object"] and \
               calculate_iou(obj["box"], filtered_obj["box"]) > iou_threshold:
                duplicate = True
                break
        
        # If not duplicate, add to filtered list
        if not duplicate:
            filtered_objects.append(obj)
    
    return filtered_objects

def process_video_keyframes(
    video_dir: str,
    caption_file: str,
    grounding_dino: GroundingDINO,
    output_file: str = None
) -> Dict[str, Any]:
    """Process all keyframes for a single video"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load captions
    try:
        with open(caption_file, 'r', encoding='utf-8') as f:
            captions = json.load(f)
        print(f"Processing {len(captions)} keyframes from {os.path.basename(caption_file)}")
    except Exception as e:
        print(f"Error reading caption file {caption_file}: {e}")
        return {"total": 0, "items": []}
    
    # Initialize list to store results
    detection_results = []
    
    # Process each keyframe
    for item in captions:
        keyframe_name = item["keyframe"]
        caption = item["caption"]
        
        # Full path to the keyframe file
        keyframe_path = os.path.join(video_dir, keyframe_name)
        
        if not os.path.exists(keyframe_path):
            print(f"Warning: Keyframe file not found at {keyframe_path}")
            continue
        
        # Extract prompt segments from caption
        prompts = extract_objects_from_caption(caption)
        
        # Initialize results for current keyframe
        keyframe_results = {
            "keyframe": keyframe_name,
            "caption": caption,
            "objects": []
        }
        
        # Detect objects using each prompt
        for prompt in prompts:
            # Detect objects
            results = grounding_dino.detect_objects(
                keyframe_path, 
                f"Find {prompt}",
                box_threshold=0.35,
                text_threshold=0.25
            )
            
            # Add results to the list
            for i, (box, score) in enumerate(zip(results["boxes"], results["scores"])):
                label = results["labels"][i] if i < len(results["labels"]) else prompt
                keyframe_results["objects"].append({
                    "prompt": prompt,
                    "object": label,
                    "box": box,
                    "score": score
                })
        
        # Apply filter_objects to remove duplicate objects and filter by score
        keyframe_results["objects"] = filter_objects(keyframe_results["objects"])
        
        print(f"Keyframe {keyframe_name}: {len(keyframe_results['objects'])} objects after filtering")
        
        # Add keyframe results to the main list
        detection_results.append(keyframe_results)
    
    # Save results to JSON file if output_file is provided
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detection_results, f, ensure_ascii=False, indent=4)
        
        print(f"Saved detection results to {output_file}")
    
    return {"total": len(detection_results), "items": detection_results}

def process_lesson(
    lesson_keyframe_dir: str,
    lesson_caption_dir: str,
    lesson_output_dir: str,
    grounding_dino: GroundingDINO
) -> Dict[str, Any]:
    """Process all videos in a lesson"""
    # Create output directory
    os.makedirs(lesson_output_dir, exist_ok=True)
    
    # Get all video directories in the lesson
    video_dirs = sorted(glob.glob(os.path.join(lesson_keyframe_dir, "V*")))
    
    # Process each video
    all_results = []
    for video_dir in video_dirs:
        video_name = os.path.basename(video_dir)
        lesson_name = os.path.basename(lesson_keyframe_dir)
        
        # Path to the caption file
        caption_file = os.path.join(lesson_caption_dir, f"{lesson_name}_{video_name}_caption.json")
        
        # Output file
        output_file = os.path.join(lesson_output_dir, f"{lesson_name}_{video_name}_detection.json")
        
        # Process video keyframes
        video_results = process_video_keyframes(
            video_dir,
            caption_file,
            grounding_dino,
            output_file
        )
        
        all_results.extend(video_results.get("items", []))
    
    return {"total": len(all_results), "items": all_results}

def detect_object(
    input_keyframe_dir: str,
    input_caption_dir: str,
    output_detection_dir: str,
    mode: str,
    lesson_name: str = None,
    video_name: str = None
) -> Dict[str, Any]:
    """Detect objects in keyframes using Grounding DINO"""
    # Create output directory
    os.makedirs(output_detection_dir, exist_ok=True)
    
    # Initialize GroundingDINO model
    grounding_dino = GroundingDINO()
    
    # Initialize results
    all_results = []
    
    if mode == "all":
        # Process all lessons
        lessons = [d for d in os.listdir(input_keyframe_dir) 
                  if os.path.isdir(os.path.join(input_keyframe_dir, d))]
        
        for lesson in lessons:
            lesson_results = process_lesson(
                os.path.join(input_keyframe_dir, lesson),
                os.path.join(input_caption_dir, lesson),
                os.path.join(output_detection_dir, lesson),
                grounding_dino
            )
            all_results.extend(lesson_results.get("items", []))
            
    elif mode == "lesson":
        # Process single lesson
        lesson_results = process_lesson(
            os.path.join(input_keyframe_dir, lesson_name),
            os.path.join(input_caption_dir, lesson_name),
            os.path.join(output_detection_dir, lesson_name),
            grounding_dino
        )
        all_results = lesson_results.get("items", [])
        
    elif mode == "single":
        # Process single video
        video_dir = os.path.join(input_keyframe_dir, lesson_name, video_name)
        caption_file = os.path.join(input_caption_dir, lesson_name, f"{lesson_name}_{video_name}_caption.json")
        
        # Create output directory for this lesson
        lesson_output_dir = os.path.join(output_detection_dir, lesson_name)
        os.makedirs(lesson_output_dir, exist_ok=True)
        
        output_file = os.path.join(lesson_output_dir, f"{lesson_name}_{video_name}_detection.json")
        
        # Process the video
        video_results = process_video_keyframes(
            video_dir,
            caption_file,
            grounding_dino,
            output_file
        )
        all_results = video_results.get("items", [])
    
    # Create summary
    summary = {
        "total": len(all_results),
        "items": all_results
    }
    
    # Save summary if there are results
    if all_results:
        summary_file = os.path.join(output_detection_dir, "detection_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)
    
    return summary