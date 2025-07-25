"""
GroundingDINO model implementation for AIO-AIClosers
Adapted from CaptionandDetection project for object detection search
"""

import os
import torch
import cv2
import numpy as np
from PIL import Image
import json
from typing import List, Dict, Tuple, Optional
from app.config import DEVICE

class GroundingDINO:
    """
    GroundingDINO model wrapper for object detection and search.
    Provides unified interface compatible with AIO-AIClosers architecture.
    """
    
    def __init__(self, device: str = DEVICE):
        """
        Initialize GroundingDINO model.
        
        Args:
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model = None
        self.model_loaded = False
        
        # Model configuration (based on CaptionandDetection structure)
        self.model_config_path = "models/weights/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.model_checkpoint_path = "models/weights/groundingdino_swint_ogc.pth"
        
        # Detection thresholds
        self.box_threshold = 0.35
        self.text_threshold = 0.25
        
        # Load model on initialization
        self._load_model()
    
    def _load_model(self):
        """Load GroundingDINO model and weights."""
        try:
            # Check if model files exist
            if not os.path.exists(self.model_config_path):
                print(f"[WARNING] GroundingDINO config not found at {self.model_config_path}")
                print("Please run: python download_groundingdino_weights.py")
                return
                
            if not os.path.exists(self.model_checkpoint_path):
                print(f"[WARNING] GroundingDINO checkpoint not found at {self.model_checkpoint_path}")
                print("Please run: python download_groundingdino_weights.py")
                return
            
            # Import GroundingDINO modules
            from groundingdino.util.inference import load_model, load_image, predict
            
            # Store inference functions
            self.load_image = load_image
            self.predict = predict
            
            # Load model
            self.model = load_model(self.model_config_path, self.model_checkpoint_path)
            self.model.to(self.device)
            self.model_loaded = True
            
            print(f"[INFO] GroundingDINO model loaded successfully on {self.device}")
            
        except ImportError as e:
            print(f"[WARNING] GroundingDINO dependencies not found: {e}")
            print("Please install: pip install groundingdino-py")
            self.model_loaded = False
        except FileNotFoundError as e:
            print(f"[WARNING] GroundingDINO model files not found: {e}")
            print("Please run: python download_groundingdino_weights.py")
            self.model_loaded = False
        except Exception as e:
            print(f"[WARNING] Failed to load GroundingDINO model: {e}")
            self.model_loaded = False
    
    def detect_objects(self, image_path: str, text_prompt: str) -> Dict:
        """
        Detect objects in image using text prompt.
        Based on CaptionandDetection implementation.
        
        Args:
            image_path (str): Path to image file
            text_prompt (str): Text description of objects to detect
            
        Returns:
            Dict: Detection results with boxes, scores, labels
        """
        if not self.model_loaded:
            return {"boxes": [], "scores": [], "labels": []}
        
        try:
            # Load and preprocess image
            image_source, image = self.load_image(image_path)
            
            # Format prompt like CaptionandDetection: "Find {object_name}"
            formatted_prompt = f"Find {text_prompt}" if not text_prompt.lower().startswith('find') else text_prompt
            
            # Perform detection
            boxes, logits, phrases = self.predict(
                model=self.model,
                image=image,
                caption=formatted_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device
            )
            
            # Convert to standard format (same as CaptionandDetection)
            H, W, _ = image_source.shape
            boxes_xyxy = boxes * torch.Tensor([W, H, W, H])
            boxes_xyxy = boxes_xyxy.cpu().numpy().tolist()
            
            return {
                "boxes": boxes_xyxy,
                "scores": logits.cpu().numpy().tolist(),
                "labels": phrases
            }
            
        except Exception as e:
            print(f"[ERROR] Detection failed for {os.path.basename(image_path)}: {e}")
            return {"boxes": [], "scores": [], "labels": []}
    
    def _create_detection_prompt(self, query: str) -> str:
        """
        Create optimized detection prompt from user query.
        
        Args:
            query (str): User's search query
            
        Returns:
            str: Optimized prompt for GroundingDINO
        """
        # Extract key objects and concepts from query
        # This can be enhanced with NLP processing
        
        # Simple approach: use query as-is with "Find" prefix
        if not query.lower().startswith(('find', 'detect', 'locate')):
            return f"Find {query}"
        return query
    
    def search(self, query: str, image_paths: List[str], topK: int = 10) -> List[str]:
        """
        Search for images containing objects matching the query.
        Enhanced with better error handling and progress tracking.
        
        Args:
            query (str): Search query describing objects to find
            image_paths (List[str]): List of image paths to search
            topK (int): Number of top results to return
            
        Returns:
            List[str]: List of image paths ranked by relevance
        """
        if not self.model_loaded:
            print("[WARNING] GroundingDINO model not loaded, returning empty results")
            return []
        
        results = []
        processed = 0
        
        for image_path in image_paths:
            try:
                # Check if image file exists
                if not os.path.exists(image_path):
                    continue
                    
                # Detect objects in image
                detection = self.detect_objects(image_path, query)
                
                # Calculate relevance score
                relevance = self._calculate_relevance(detection, query)
                
                if relevance > 0:
                    results.append((image_path, relevance))
                
                processed += 1
                if processed % 100 == 0:
                    print(f"[INFO] Processed {processed}/{len(image_paths)} images")
                    
            except Exception as e:
                print(f"[WARNING] Error processing {os.path.basename(image_path)}: {e}")
                continue
        
        # Sort by relevance and return top K
        results.sort(key=lambda x: x[1], reverse=True)
        return [path for path, _ in results[:topK]]
    
    def _get_keyframe_files(self, keyframes_path: str) -> List[str]:
        """Get list of keyframe files from directory."""
        keyframe_files = []
        
        # Walk through directory structure (L01/V001/*.jpg)
        for root, dirs, files in os.walk(keyframes_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    keyframe_files.append(os.path.join(root, file))
        
        return sorted(keyframe_files)
    
    def _calculate_relevance(self, detection: Dict, query: str) -> float:
        """
        Calculate relevance score based on detection results.
        Enhanced scoring considering multiple factors.
        
        Args:
            detection (Dict): Detection results from detect_objects
            query (str): Original search query
            
        Returns:
            float: Relevance score (0.0 to 1.0)
        """
        if not detection["scores"] or not detection["boxes"]:
            return 0.0
        
        # Filter by confidence threshold
        valid_scores = [score for score in detection["scores"] if score >= self.box_threshold]
        
        if not valid_scores:
            return 0.0
        
        # Calculate base score from maximum confidence
        max_score = max(valid_scores)
        
        # Bonus for multiple detections (indicates strong presence)
        detection_count = len(valid_scores)
        count_bonus = min(0.1 * (detection_count - 1), 0.3)  # Max 30% bonus
        
        # Combine scores
        final_score = min(max_score + count_bonus, 1.0)
        
        return final_score
    
    def text_search(self, query: str, top_k: int = 100) -> Tuple[List[float], List[float], List[str]]:
        """
        Text search interface compatible with AIO-AIClosers FAISS handlers.
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            
        Returns:
            Tuple[List[float], List[float], List[str]]: distances, similarities, paths
        """
        # Import config to get keyframes path
        from app.config import KEYFRAMES_FOLDER
        
        # Perform search
        paths = self.search(query, self._get_keyframe_files(KEYFRAMES_FOLDER), top_k)
        
        # Create mock distances and similarities for compatibility
        # In a real implementation, these would be actual detection scores
        similarities = [1.0 - (i * 0.01) for i in range(len(paths))]  # Decreasing similarity
        distances = [i * 0.01 for i in range(len(paths))]  # Increasing distance
        
        return distances, similarities, paths
    
    def get_detection_summary(self, image_path: str, query: str) -> Dict:
        """
        Get detailed detection summary for an image.
        Enhanced with more comprehensive information.
        
        Args:
            image_path (str): Path to image file
            query (str): Search query
            
        Returns:
            Dict: Detailed detection information
        """
        if not self.model_loaded:
            return {
                "image_path": image_path,
                "query": query,
                "error": "GroundingDINO model not loaded",
                "detection_count": 0,
                "max_confidence": 0.0,
                "relevance_score": 0.0,
                "detections": {"boxes": [], "scores": [], "labels": []}
            }
        
        detection = self.detect_objects(image_path, query)
        relevance = self._calculate_relevance(detection, query)
        
        # Filter high-confidence detections
        high_conf_detections = [
            (box, score, label) 
            for box, score, label in zip(detection["boxes"], detection["scores"], detection["labels"])
            if score >= self.box_threshold
        ]
        
        return {
            "image_path": image_path,
            "query": query,
            "detection_count": len(detection["boxes"]),
            "high_confidence_count": len(high_conf_detections),
            "max_confidence": max(detection["scores"]) if detection["scores"] else 0.0,
            "avg_confidence": sum(detection["scores"]) / len(detection["scores"]) if detection["scores"] else 0.0,
            "relevance_score": relevance,
            "detections": detection,
            "model_loaded": self.model_loaded
        }


def download_model_weights():
    """
    Download GroundingDINO model weights if not present.
    This function should be called during setup.
    """
    import urllib.request
    import os
    
    # Create weights directory
    weights_dir = "models/weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    # Model files URLs
    config_url = "https://github.com/IDEA-Research/GroundingDINO/raw/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    checkpoint_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    
    config_path = os.path.join(weights_dir, "GroundingDINO_SwinT_OGC.py")
    checkpoint_path = os.path.join(weights_dir, "groundingdino_swint_ogc.pth")
    
    # Download config file
    if not os.path.exists(config_path):
        print("Downloading GroundingDINO config...")
        urllib.request.urlretrieve(config_url, config_path)
        print(f"Config downloaded to {config_path}")
    
    # Download checkpoint (large file)
    if not os.path.exists(checkpoint_path):
        print("Downloading GroundingDINO checkpoint (this may take a while)...")
        urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
        print(f"Checkpoint downloaded to {checkpoint_path}")
    
    print("GroundingDINO model weights ready!")


if __name__ == "__main__":
    # Test the model
    grounding = GroundingDINO()
    if grounding.model_loaded:
        print("GroundingDINO model loaded successfully!")
    else:
        print("Failed to load GroundingDINO model. Please check installation and weights.")
