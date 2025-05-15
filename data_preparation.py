import os
import faiss
import numpy as np
import pickle
from typing import List, Dict, Tuple, Any
from PIL import Image
import time
from tqdm import tqdm

from vlm import BaseVLM

def build_faiss_index(
    image_paths: List[str], 
    model: BaseVLM, 
    output_dir: str = "faiss_index",
    batch_size: int = 32,
    verbose: bool = True
) -> Tuple[faiss.Index, Dict[int, str]]:
    """Build a FAISS index from a list of image paths using a VLM model.
    
    Args:
        image_paths (List[str]): List of paths to images to index
        model (BaseVLM): Vision-Language Model to use for encoding images
        output_dir (str): Directory to save the index and id-to-path mapping
        batch_size (int): Batch size for processing images
        verbose (bool): Whether to show progress bars and logs
        
    Returns:
        Tuple[faiss.Index, Dict[int, str]]: The FAISS index and id-to-path mapping
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Create a dictionary mapping IDs to image paths
    id2path = {i: path for i, path in enumerate(image_paths)}
    
    # Calculate embeddings for all images
    if verbose:
        print(f"Computing embeddings for {len(image_paths)} images...")
    
    embeddings = []
    
    # Process images in batches
    for i in tqdm(range(0, len(image_paths), batch_size), disable=not verbose):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        # Load images
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                batch_images.append(img)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                # Use a placeholder instead
                batch_images.append(Image.new('RGB', (224, 224), color='black'))
        
        # Encode images
        batch_embeddings = model.encode_batch_images(batch_images)
        embeddings.append(batch_embeddings)
        
        # Close images to save memory
        for img in batch_images:
            img.close()
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(embeddings).astype(np.float32)
    
    if verbose:
        print(f"Building FAISS index with dimension {all_embeddings.shape[1]}...")
    
    # Create and train the index
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product similarity (cosine)
    index.add(all_embeddings)
    
    # Save the index and ID mapping
    index_path = os.path.join(output_dir, "faiss_index.bin")
    mapping_path = os.path.join(output_dir, "id2path.pkl")
    
    faiss.write_index(index, index_path)
    with open(mapping_path, 'wb') as f:
        pickle.dump(id2path, f)
    
    if verbose:
        print(f"FAISS index saved to {index_path}")
        print(f"ID to path mapping saved to {mapping_path}")
    
    return index, id2path

def extract_keyframes_from_video(
    video_path: str, 
    output_dir: str,
    interval: float = 1.0,  # Extract a frame every X seconds
    min_scene_change: float = 0.3,  # Minimum difference to consider a scene change
    max_frames: int = None  # Maximum number of frames to extract
) -> List[str]:
    """Extract keyframes from a video.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save extracted frames
        interval (float): Extract a frame every X seconds
        min_scene_change (float): Minimum difference to consider a scene change
        max_frames (int): Maximum number of frames to extract
        
    Returns:
        List[str]: List of paths to extracted keyframes
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV (cv2) is required for keyframe extraction")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interval
    frame_interval = int(fps * interval)
    
    keyframe_paths = []
    prev_frame = None
    frame_count = 0
    
    while True:
        ret, frame = video.read()
        
        if not ret:
            break
            
        frame_count += 1
        
        # Check if this frame should be considered (based on interval)
        if frame_count % frame_interval != 0:
            continue
            
        # Convert to grayscale for scene change detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # If we have a previous frame, check for scene change
        save_frame = False
        if prev_frame is None:
            save_frame = True  # Always save the first frame
        else:
            # Calculate difference between frames
            diff = cv2.absdiff(prev_frame, gray)
            non_zero_count = np.count_nonzero(diff)
            change_percent = non_zero_count / diff.size
            
            # Save if it's a scene change
            if change_percent > min_scene_change:
                save_frame = True
                
        prev_frame = gray
        
        if save_frame:
            # Generate filename based on timestamp
            timestamp = frame_count / fps
            filename = f"{os.path.basename(video_path).split('.')[0]}_frame_{int(timestamp)}.jpg"
            output_path = os.path.join(output_dir, filename)
            
            # Save the frame
            cv2.imwrite(output_path, frame)
            keyframe_paths.append(output_path)
            
        # Check if we have enough frames
        if max_frames and len(keyframe_paths) >= max_frames:
            break
    
    video.release()
    
    return keyframe_paths

def build_index_from_videos(
    video_directory: str,
    model: BaseVLM,
    output_dir: str = "faiss_index",
    keyframes_dir: str = "data/keyframes",
    interval: float = 1.0,
    min_scene_change: float = 0.3,
    max_frames_per_video: int = None,
    batch_size: int = 32,
    verbose: bool = True
) -> Tuple[faiss.Index, Dict[int, str]]:
    """Extract keyframes from videos and build a FAISS index.
    
    Args:
        video_directory (str): Directory containing video files
        model (BaseVLM): Vision-Language Model to use for encoding images
        output_dir (str): Directory to save the index and id-to-path mapping
        keyframes_dir (str): Directory to save extracted keyframes
        interval (float): Extract a frame every X seconds
        min_scene_change (float): Minimum difference to consider a scene change
        max_frames_per_video (int): Maximum number of frames to extract per video
        batch_size (int): Batch size for processing images
        verbose (bool): Whether to show progress bars and logs
        
    Returns:
        Tuple[faiss.Index, Dict[int, str]]: The FAISS index and id-to-path mapping
    """
    # Create keyframes directory if it doesn't exist
    if not os.path.exists(keyframes_dir):
        os.makedirs(keyframes_dir)
        
    all_keyframe_paths = []
    
    # Get all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_files = []
    
    for root, _, files in os.walk(video_directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
    
    if verbose:
        print(f"Found {len(video_files)} videos in {video_directory}")
        
    # Extract keyframes from each video
    for video_path in tqdm(video_files, desc="Extracting keyframes", disable=not verbose):
        video_name = os.path.basename(video_path).split('.')[0]
        video_keyframes_dir = os.path.join(keyframes_dir, video_name)
        
        keyframe_paths = extract_keyframes_from_video(
            video_path=video_path,
            output_dir=video_keyframes_dir,
            interval=interval,
            min_scene_change=min_scene_change,
            max_frames=max_frames_per_video
        )
        
        all_keyframe_paths.extend(keyframe_paths)
        
        if verbose:
            print(f"Extracted {len(keyframe_paths)} keyframes from {video_path}")
    
    # Build FAISS index from extracted keyframes
    return build_faiss_index(
        image_paths=all_keyframe_paths,
        model=model,
        output_dir=output_dir,
        batch_size=batch_size,
        verbose=verbose
    ) 