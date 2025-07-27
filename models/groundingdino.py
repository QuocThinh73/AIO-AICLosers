import os
import sys
import subprocess
import urllib.request
from typing import Dict, List, Union, Tuple
from PIL import Image

class GroundingDINO:
    def __init__(self, device=None):
        """
        Initialize Grounding DINO model
        Args:
            device: Device to run inference on ('cpu' or 'cuda')
        """
        # Set model paths and create directories
        self._setup_model_paths()
        
        # Initialize model
        self.model = None
        
        # Set device
        try:
            import torch
            self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        except:
            self.device = "cpu"
    
    def _install_dependencies(self):
        """
        Simple dependency installation similar to Kaggle notebook
        """
        print("Installing required packages...")
        try:
            # Use specific commands similar to the Kaggle notebook
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "groundingdino-py"],
                stdout=subprocess.PIPE
            )
            print("✓ Dependencies installed")
        except Exception as e:
            print(f"Warning: Error installing dependencies: {e}")
    
    def _setup_model_paths(self):
        """Set up paths for model configuration and checkpoint files"""
        # Create directories if they don't exist
        os.makedirs("GroundingDINO/groundingdino/config", exist_ok=True)
        os.makedirs("weights", exist_ok=True)
        
        # Set file paths
        self.config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.checkpoint_path = "weights/groundingdino_swint_ogc.pth"
    
    def _download_config(self):
        """Download the model configuration file if not already present"""
        if os.path.exists(self.config_path):
            print(f"Config file already exists at {self.config_path}")
            return
        
        print(f"Downloading config file to {self.config_path}...")
        config_url = "https://github.com/IDEA-Research/GroundingDINO/raw/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        
        try:
            urllib.request.urlretrieve(config_url, self.config_path)
            print("Config file downloaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to download config file: {e}")
    
    def _download_checkpoint(self):
        """Download the model checkpoint file if not already present"""
        if os.path.exists(self.checkpoint_path):
            print(f"Checkpoint file already exists at {self.checkpoint_path}")
            return
        
        print(f"Downloading checkpoint file to {self.checkpoint_path}...")
        checkpoint_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        
        try:
            urllib.request.urlretrieve(checkpoint_url, self.checkpoint_path)
            print("Checkpoint downloaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to download checkpoint file: {e}")
    

    
    def load_model(self):
        """
        Load the Grounding DINO model, downloading and setting up everything as needed
        """
        # Check if model is already loaded
        if self.model is not None:
            return
        
        # Install dependencies first
        self._install_dependencies()
        
        # Download model config and checkpoint files
        self._download_config()
        self._download_checkpoint()
        
        print("\nLoading Grounding DINO model...")
        try:
            # Follow exact import sequence from Kaggle notebook
            import torch
            from groundingdino.util.inference import load_model
            
            # Load the model with exact same call as in notebook
            self.model = load_model(self.config_path, self.checkpoint_path)
            self.model = self.model.to(self.device)
            
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load Grounding DINO model: {e}")
    
    def detect_objects(self, 
                      image: Union[str, Image.Image], 
                      text_prompt: str, 
                      box_threshold: float = 0.35, 
                      text_threshold: float = 0.25) -> Dict:
        """
        Detect objects in an image using text prompts
        
        Args:
            image: Image to process (file path or PIL Image)
            text_prompt: Text prompt for object detection
            box_threshold: Threshold for box confidence
            text_threshold: Threshold for text confidence
            
        Returns:
            Dict with detection results (boxes, labels, scores)
        """
        # Ensure model is loaded (will download and set up if needed)
        if self.model is None:
            print("Loading model first...")
            self.load_model()
        
        print(f"Detecting objects in image with prompt: '{text_prompt}'")
        try:
            # Import modules inside function to avoid circular imports
            import torch
            import numpy as np
            from groundingdino.util.inference import load_image, predict
                
            # Convert path to PIL Image if needed
            if isinstance(image, str):
                print(f"Loading image from path: {image}")
                image_source, image_tensor = load_image(image)
            else:
                image_source = np.array(image)
                image_tensor = image
                
            print(f"Running inference with box_threshold={box_threshold}, text_threshold={text_threshold}")
            # Run inference
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.device
            )
            
            print(f"Raw prediction results: {len(boxes)} boxes, {len(logits) if logits is not None else 0} scores, {len(phrases) if phrases is not None else 0} phrases")
            
            # Convert to numpy for consistency
            if isinstance(boxes, torch.Tensor) and len(boxes) > 0:
                H, W, _ = image_source.shape
                boxes_xyxy = boxes * torch.tensor([W, H, W, H], device=boxes.device)
                boxes = boxes_xyxy.cpu().numpy().tolist()
            else:
                boxes = [] if not isinstance(boxes, list) else boxes
                
            if isinstance(logits, torch.Tensor):
                scores = logits.cpu().numpy().tolist()
            else:
                scores = logits if logits is not None else []
                
        except Exception as e:
            print(f"Error during object detection: {e}")
            import traceback
            traceback.print_exc()
            return {"boxes": [], "scores": [], "labels": []}
            
        # Format results
        results = {
            "boxes": boxes,
            "scores": scores,
            "labels": phrases if phrases is not None else []
        }
        
        print(f"Final detection results: {len(results['boxes'])} boxes, {len(results['scores'])} scores, {len(results['labels'])} labels")
        return results