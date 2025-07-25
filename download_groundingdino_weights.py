#!/usr/bin/env python3
"""
Download GroundingDINO model weights and config files
Based on CaptionandDetection implementation
"""

import os
import urllib.request
from pathlib import Path

def download_file(url, filepath):
    """Download file from URL to filepath."""
    print(f"Downloading {os.path.basename(filepath)}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"[OK] Downloaded: {filepath}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {filepath}: {e}")
        return False

def download_groundingdino_weights():
    """Download GroundingDINO model config and weights."""
    
    # Create directories
    weights_dir = Path("models/weights")
    config_dir = weights_dir / "GroundingDINO" / "groundingdino" / "config"
    
    weights_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading GroundingDINO model files...")
    print("=" * 60)
    
    # URLs from CaptionandDetection
    config_url = "https://github.com/IDEA-Research/GroundingDINO/raw/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    weights_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    
    # File paths
    config_path = config_dir / "GroundingDINO_SwinT_OGC.py"
    weights_path = weights_dir / "groundingdino_swint_ogc.pth"
    
    # Download files
    success = True
    
    # Download config file
    if not config_path.exists():
        success &= download_file(config_url, config_path)
    else:
        print(f"[OK] Config already exists: {config_path}")
    
    # Download weights file (this is large ~694MB)
    if not weights_path.exists():
        print(f"Downloading model weights (~694MB)...")
        success &= download_file(weights_url, weights_path)
    else:
        print(f"[OK] Weights already exist: {weights_path}")
    
    print("=" * 60)
    if success:
        print("GroundingDINO model files downloaded successfully!")
        print(f"Config: {config_path}")
        print(f"Weights: {weights_path}")
        print("\nYou can now run the integration tests:")
        print("python test_integration.py")
    else:
        print("[ERROR] Some downloads failed. Please check your internet connection.")
    
    return success

if __name__ == "__main__":
    download_groundingdino_weights()
