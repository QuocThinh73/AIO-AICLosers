import os
import torch
import numpy as np
from typing import Dict, List, Union, Tuple
from PIL import Image
import subprocess
import sys
import gdown

class GroundingDINO:
    def __init__(self, device=None):
        """
        Initialize Grounding DINO model
        Args:
            device: Device to run inference on ('cpu' or 'cuda')
        """
        # Install required packages first to ensure all imports work
        self._check_and_install_packages()
        
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        # Initialize model paths
        self._setup_model_paths()
    
    def _check_and_install_packages(self):
        """
        Check and install required packages if they are not already installed
        """
        required_packages = [
            "torch", "torchvision", "opencv-python", "matplotlib", "transformers>=4.25.1", 
            "timm>=0.6.13", "supervision", "pycocotools", "gdown", "addict", "yapf", 
            "terminaltables", "tqdm"
        ]
        
        try:
            import importlib
            import pip
            
            packages_to_install = []
            for package in required_packages:
                # Extract package name without version specifier
                package_name = package.split('>=')[0].split('==')[0].strip()
                try:
                    importlib.import_module(package_name)
                    print(f"✓ {package_name} is already installed")
                except ImportError:
                    packages_to_install.append(package)
            
            if packages_to_install:
                print(f"Installing missing packages: {', '.join(packages_to_install)}")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-input", "--quiet"] + packages_to_install)
                print("All required packages installed successfully")
        except Exception as e:
            print(f"Warning: Could not check or install packages automatically: {e}")
            print("Continuing with existing packages...")
    
    def _setup_model_paths(self):
        """Set up paths for model configuration and checkpoint files"""
        # Create directories if they don't exist
        os.makedirs("GroundingDINO", exist_ok=True)
        os.makedirs("weights", exist_ok=True)
        
        # Set file paths
        self.config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.checkpoint_path = "weights/groundingdino_swint_ogc.pth"
    
    def _clone_groundingdino(self):
        """Clone the GroundingDINO repository if not already cloned"""
        if os.path.exists("GroundingDINO"):
            print("GroundingDINO repository already exists")
            return
        
        print("Cloning GroundingDINO repository...")
        try:
            subprocess.run(["git", "clone", "https://github.com/IDEA-Research/GroundingDINO.git"], check=True)
            print("Repository cloned successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to clone repository: {e}")
    
    def _download_checkpoint(self):
        """Download the model checkpoint file if not already present"""
        if os.path.exists(self.checkpoint_path):
            print(f"Checkpoint file already exists at {self.checkpoint_path}")
            return
        
        print(f"Downloading checkpoint file to {self.checkpoint_path}...")
        
        # Download from Google Drive using ID
        checkpoint_id = "1jjkRSa7aLsLMx0rUx-fT1g6Q9H1XN_1T"
        try:
            gdown.download(id=checkpoint_id, output=self.checkpoint_path, quiet=False)
            print("Checkpoint downloaded successfully")
        except Exception as e:
            print(f"Google Drive download failed: {e}, trying direct URL...")
            
            # Fallback to direct URL if Google Drive fails
            direct_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
            try:
                import urllib.request
                urllib.request.urlretrieve(direct_url, self.checkpoint_path)
                print("Checkpoint downloaded successfully from direct URL")
            except Exception as e:
                raise RuntimeError(f"Failed to download checkpoint file: {e}")
    
    def _install_dependencies(self):
        """Add GroundingDINO to Python path and verify module structure"""
        try:
            # Add GroundingDINO to Python path instead of installing it
            cwd = os.getcwd()
            print(f"Current working directory: {cwd}")
            
            # Try multiple possible paths for Kaggle and local environments
            possible_paths = [
                os.path.abspath("GroundingDINO"),
                os.path.join(cwd, "GroundingDINO"),
                os.path.join(os.path.dirname(cwd), "GroundingDINO"),
            ]
            
            groundingdino_path = None
            for path in possible_paths:
                if os.path.exists(path) and os.path.isdir(path):
                    groundingdino_path = path
                    if path not in sys.path:
                        sys.path.insert(0, path)
                    print(f"Added {path} to Python path")
                    break
            
            if groundingdino_path is None:
                print("Warning: Could not find GroundingDINO directory")
                return
            
            # Check if the module structure exists
            module_path = os.path.join(groundingdino_path, "groundingdino")
            if not os.path.exists(module_path):
                print(f"Warning: groundingdino module not found at {module_path}")
                # List directory contents for debugging
                print(f"GroundingDINO directory contents: {os.listdir(groundingdino_path)}")
            else:
                print(f"Found groundingdino module at {module_path}")
                
                # Explicitly add the 'groundingdino' module path
                if module_path not in sys.path:
                    sys.path.insert(0, module_path)
                    print(f"Added {module_path} to Python path")
            
            # Print sys.path for debugging
            print(f"Python path (first 3 entries): {sys.path[:3]}")
        except Exception as e:
            print(f"Warning: Error configuring GroundingDINO paths: {e}")
            print("Attempting to continue anyway...")
            import traceback
            traceback.print_exc()
    
    def load_model(self):
        """
        Load the Grounding DINO model, downloading and setting up everything as needed
        """
        # Check if model is already loaded
        if self.model is not None:
            return
        
        # Clone repository, download checkpoint, and add to Python path
        self._clone_groundingdino()
        self._download_checkpoint()
        self._install_dependencies()
        
        # Now import and load the model
        try:
            
            from groundingdino.util.inference import load_model
            
            # Load the model
            self.model = load_model(self.config_path, self.checkpoint_path)
            self.model.to(self.device)
            print(f"Grounding DINO model loaded successfully on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Error loading Grounding DINO model: {e}")
    
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
            self.load_model()
            
        # Import here to ensure dependencies are installed
        from groundingdino.util.inference import load_image, predict
            
        # Convert path to PIL Image if needed
        if isinstance(image, str):
            image_source, image = load_image(image)
        else:
            import numpy as np
            image_source = np.array(image)
            
        # Run inference
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_source, 
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )
        
        # Convert to numpy for consistency
        if isinstance(boxes, torch.Tensor):
            H, W, _ = image_source.shape
            boxes_xyxy = boxes * torch.Tensor([W, H, W, H])
            boxes = boxes_xyxy.cpu().numpy().tolist()
        if isinstance(logits, torch.Tensor):
            scores = logits.cpu().numpy().tolist()
        else:
            scores = logits
            
        # Format results
        results = {
            "boxes": boxes,
            "scores": scores,
            "labels": phrases
        }
        
        return results