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
import shutil
from tqdm import tqdm

# Import our model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.internvl3 import InternVL3


def get_captioning_model():
    _ensure_dependencies()
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.internvl3 import InternVL3
    
    model = InternVL3(task="image_captioning", use_quantization=True)
    print(f"Initialized InternVL3 model on {model.device} device")
    return model

def process_video(video_dir, output_dir, lesson_name, video_name, model):
    video_results = model.process_video(video_dir)
    
    lesson_output_dir = os.path.join(output_dir, lesson_name)
    os.makedirs(lesson_output_dir, exist_ok=True)
    
    output_file = os.path.join(lesson_output_dir, f"{lesson_name}_{video_name}_caption.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(video_results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_file}")

def generate_captions(input_dir, output_dir, mode, lesson_name=None, video_name=None):
    os.makedirs(output_dir, exist_ok=True)
    
    model = get_captioning_model()
    
    if mode == "single":
        video_dir = os.path.join(input_dir, lesson_name, video_name)
        process_video(video_dir, output_dir, lesson_name, video_name, model)
    elif mode == "lesson":
        lesson_dir = os.path.join(input_dir, lesson_name)
        for video_folder in sorted(os.listdir(lesson_dir)):
            video_dir = os.path.join(lesson_dir, video_folder)
            if os.path.isdir(video_dir):
                process_video(video_dir, output_dir, lesson_name, video_folder, model)
    else:
        for lesson_folder in sorted(os.listdir(input_dir)):
            lesson_dir = os.path.join(input_dir, lesson_folder)
            if os.path.isdir(lesson_dir):
                for video_folder in sorted(os.listdir(lesson_dir)):
                    video_dir = os.path.join(lesson_dir, video_folder)
                    if os.path.isdir(video_dir):
                        process_video(video_dir, output_dir, lesson_folder, video_folder, model)

