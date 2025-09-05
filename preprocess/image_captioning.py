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
from transformers import BitsAndBytesConfig
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import time
import gc

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

# Constants for image preprocessing (from kaggle_internvl_official_batch.py)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size=448):
    """Build image transformation pipeline (from kaggle_internvl_official_batch.py)"""
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from target ratios (from kaggle_internvl_official_batch.py)"""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
                
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    """Dynamic preprocessing with aspect ratio consideration (from kaggle_internvl_official_batch.py)"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
        
    return processed_images

def load_image_batch(image_file, input_size=448, max_num=12):
    """Load and preprocess image for batch processing (from kaggle_internvl_official_batch.py)"""
    if isinstance(image_file, str):
        image = Image.open(image_file).convert('RGB')
    else:
        image = image_file.convert('RGB')
        
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    
    return pixel_values

def batch_process_images(model, image_list, batch_size=8):
    """
    Process images using batch processing for faster inference
    
    Args:
        model: InternVL35 model instance
        image_list: List of image file paths
        batch_size: Number of images to process in each batch
    
    Returns:
        List of captions
    """
    print(f"🔄 Processing {len(image_list)} images in batches of {batch_size}...")
    
    all_captions = []
    
    for i in tqdm(range(0, len(image_list), batch_size), desc="Processing batches"):
        batch_images = image_list[i:i+batch_size]
        
        # Load and process images for batch
        pixel_values_list = []
        num_patches_list = []
        
        for img_path in batch_images:
            pixel_values = load_image_batch(img_path, max_num=12).to(torch.bfloat16)
            if torch.cuda.is_available():
                pixel_values = pixel_values.cuda()
            
            pixel_values_list.append(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
        
        # Concatenate all images for true batch processing
        combined_pixel_values = torch.cat(pixel_values_list, dim=0)
        
        # Create identical questions for each image
        questions = [f"<image>\n{model.prompt}"] * len(batch_images)
        
        # Generation config for consistent output
        generation_config = dict(
            max_new_tokens=256,
            do_sample=True,
            temperature=0.3,
            top_p=0.9
        )
        
        # Use batch_chat for simultaneous processing
        try:
            responses = model.model.batch_chat(
                model.tokenizer,
                combined_pixel_values,
                num_patches_list=num_patches_list,
                questions=questions,
                generation_config=generation_config
            )
            
            # Clean responses and add to results
            for response in responses:
                all_captions.append(response.strip())
                
        except Exception as e:
            print(f"Warning: Batch processing failed, falling back to sequential: {e}")
            # Fallback to sequential processing for this batch
            for img_path in batch_images:
                caption = model.process_keyframe(img_path)
                all_captions.append(caption)
        
        # Clean memory after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    return all_captions

def get_captioning_model():
    """Get captioning model with 4-bit quantization"""
    _ensure_dependencies()
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        # Initialize model with 4-bit quantization (handled internally by InternVL35)
        model = InternVL35(task="image_captioning", use_quantization=True)
        print(f"✅ Initialized InternVL3.5-1B model with 4-bit quantization on {model.device} device")
        
        # Show memory usage if CUDA available
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"🧠 GPU Memory: {memory_allocated:.2f}GB")
        
        return model
    except Exception as e:
        print(f"Error initializing captioning model: {e}")
        import traceback
        traceback.print_exc()
        raise



def process_video(video_dir, output_dir, lesson_name, video_name, model, batch_size=8):
    # Process all keyframes in a video directory using batch processing
    keyframes = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
    
    if not keyframes:
        print(f"Warning: No keyframes found in {video_dir}")
        return
    
    print(f"🔄 Processing {len(keyframes)} keyframes for {lesson_name}/{video_name} with batch_size={batch_size}")
    
    # Use batch processing for faster inference
    captions = batch_process_images(model, keyframes, batch_size=batch_size)
    
    # Format results
    video_results = []
    for keyframe_path, caption in zip(keyframes, captions):
        keyframe_name = os.path.basename(keyframe_path)
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

def generate_captions(input_dir, output_dir, mode, lesson_name=None, video_name=None, batch_size=8):
    os.makedirs(output_dir, exist_ok=True)
    
    model = get_captioning_model()
    result = {"status": "success", "message": "Caption generation completed successfully with batch processing"}
    
    print(f"🚀 Using batch processing with batch_size={batch_size} for optimized inference")
    
    try:
        if mode == "single":
            video_dir = os.path.join(input_dir, lesson_name, video_name)
            process_video(video_dir, output_dir, lesson_name, video_name, model, batch_size)
        elif mode == "lesson":
            lesson_dir = os.path.join(input_dir, lesson_name)
            for video_folder in sorted(os.listdir(lesson_dir)):
                video_dir = os.path.join(lesson_dir, video_folder)
                if os.path.isdir(video_dir):
                    process_video(video_dir, output_dir, lesson_name, video_folder, model, batch_size)
        else:
            for lesson_folder in sorted(os.listdir(input_dir)):
                lesson_dir = os.path.join(input_dir, lesson_folder)
                if os.path.isdir(lesson_dir):
                    for video_folder in sorted(os.listdir(lesson_dir)):
                        video_dir = os.path.join(lesson_dir, video_folder)
                        if os.path.isdir(video_dir):
                            process_video(video_dir, output_dir, lesson_folder, video_folder, model, batch_size)
        
        return result
    except Exception as e:
        print(f"Error generating captions: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

