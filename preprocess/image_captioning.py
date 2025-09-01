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
        "transformers>=4.52.1",  # For InternVL3.5 model loading
        "bitsandbytes",  # For quantization
        "accelerate",   # For optimized inference
        "torch"         # PyTorch
    ]
    
    for package in required_packages:
        # Skip torch installation as it's pre-installed in most environments with correct CUDA versions
        if package == "torch":
            continue
            
        try:
            if package.startswith("transformers") and in_kaggle:
                # On Kaggle, always install latest transformers from source for InternVL3.5 support
                print(f"Installing {package} from source (required for latest InternVL3.5 support)...")
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

# Import our model using importlib to handle dot in filename
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib.util

# Load InternVL35 module dynamically to handle dot in filename
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
model_path = os.path.join(models_dir, 'internvl3.5.py')

if os.path.exists(model_path):
    # Load module using importlib.util
    spec = importlib.util.spec_from_file_location("internvl3_5_module", model_path)
    internvl3_5_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(internvl3_5_module)
    InternVL35 = internvl3_5_module.InternVL35
else:
    # Fallback to standard import
    try:
        from models.internvl3_5 import InternVL35
    except ImportError:
        from models.internvl3 import InternVL35


def get_captioning_model():
    _ensure_dependencies()
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        # Initialize the model (use quantization if CUDA is available)
        model = InternVL35(task="image_captioning", use_quantization=True)
        print(f"Initialized InternVL3.5-1B model on {model.device} device")
        return model
    except Exception as e:
        print(f"Error initializing captioning model: {e}")
        import traceback
        traceback.print_exc()
        raise



def process_video(video_dir, output_dir, lesson_name, video_name, model):
    # Process all keyframes in a video directory
    keyframes = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
    
    if not keyframes:
        print(f"Warning: No keyframes found in {video_dir}")
        return
    
    video_results = []
    for keyframe_path in tqdm(keyframes, desc=f"Processing {lesson_name}/{video_name}"):
        keyframe_name = os.path.basename(keyframe_path)
        caption = model.process_keyframe(keyframe_path)
        video_results.append({
            "keyframe": keyframe_name,
            "caption": caption
        })
    
    lesson_output_dir = os.path.join(output_dir, lesson_name)
    os.makedirs(lesson_output_dir, exist_ok=True)
    
    output_file = os.path.join(lesson_output_dir, f"{lesson_name}_{video_name}_caption.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(video_results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_file}")

def generate_captions(input_dir, output_dir, mode, lesson_name=None, video_name=None):
    os.makedirs(output_dir, exist_ok=True)
    
    model = get_captioning_model()
    result = {"status": "success", "message": "Caption generation completed successfully"}
    
    try:
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
        
        return result
    except Exception as e:
        print(f"Error generating captions: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

