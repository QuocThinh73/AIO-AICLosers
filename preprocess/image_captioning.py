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
import time
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
    from models.internvl3 import InternVL3
    
    try:
        # Initialize the model (use quantization if CUDA is available)
        model = InternVL35(task="image_captioning", use_quantization=True)
        print(f"Initialized InternVL3.5-1B model on {model.device} device")
        
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
            return {"status": "error", "message": error_msg}
            
        # Zip caption results for easy download
        zip_path = zip_caption_results(output_dir)
        
        # Return results as dictionary (similar to object_detection.py)
        return {"status": "success", "message": f"Caption generation completed successfully. Results zipped to {zip_path}"}
        
    except Exception as e:
        print(f"Error generating captions: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}



def zip_caption_results(output_dir):
    """
    Zip all caption results in the output directory for easy download
    
    Args:
        output_dir (str): Path to the output directory containing caption results
        
    Returns:
        str: Path to the created zip file
    """
    # Create a timestamp for unique zip file name
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    zip_name = f"caption_results_{timestamp}"
    zip_path = os.path.join(os.path.dirname(output_dir), zip_name)
    
    # Create zip archive
    shutil.make_archive(zip_path, 'zip', output_dir)
    
    print(f"Caption results zipped to {zip_path}.zip")
    return f"{zip_path}.zip"


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
    result = {"status": "success", "message": "Caption generation completed successfully"}
    
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
    
    # Create zip file for easy download after all processing is complete
    try:
        zip_path = zip_caption_results(output_dir)
        result = {"status": "success", "message": f"Caption generation completed successfully. Results zipped to {zip_path}"}
        print(f"✅ All captions generated and zipped to {zip_path}")
    except Exception as e:
        print(f"Warning: Failed to zip results: {e}")
        # Continue without zipping if there's an error
        
    return result

