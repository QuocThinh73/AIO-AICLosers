import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import os
import json
import glob
from tqdm import tqdm
import warnings
import requests
from io import BytesIO

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Constants for image preprocessing
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVL35:
    def __init__(self, task, model_checkpoint="OpenGVLab/InternVL3_5-4B", max_new_tokens=512, use_quantization=True):
        """
        Initialize the InternVL3.5-4B model for different tasks.
        
        Args:
            task (str): Task type - "image_captioning" or "news_anchor_classification"
            model_checkpoint (str): Hugging Face model checkpoint
            max_new_tokens (int): Maximum tokens to generate
            use_quantization (bool): Whether to use 4-bit quantization
        """
        self.task = task
        self.model_checkpoint = model_checkpoint
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_type = torch.bfloat16
        
        # Set environment variables for compatibility
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
        os.environ["DISABLE_FLASH_ATTN"] = "1"
        
        # Set task-specific prompts
        self._setup_prompt()
        self._load_model(use_quantization)
    
    def _setup_prompt(self):
        """Setup task-specific prompts."""
        if self.task == "image_captioning":
            self.prompt = """You are an image captioning assistant processing still frames from live broadcast news videos. 
Provide a concise but informative description that mentions the main subjects (people or objects), their actions, and the scene context. 
Ignore any on-screen graphics such as ticker text, news banners, program logos, watermarks, or other unrelated overlays."""
            
        elif self.task == "news_anchor_classification":
            self.prompt = """Does the image show a news anchor **actively presenting** news in a professional **broadcast TV studio** (e.g. desk with news branding, lighting rigs, large studio screens, official newsroom setup)?

Only answer YES if:
– The person is clearly delivering news (e.g. reading script, facing camera).
– The environment includes professional TV studio features.

Otherwise, answer NO.

Do not be misled by graphics or text overlays. 
Only say YES if the person is a real news anchor in a live TV studio with full broadcasting setup. 
Ignore marketing displays, showroom backgrounds, or mock studio designs.

Answer with only YES or NO."""
    
    def _load_model(self, use_quantization=True):
        """Load the tokenizer and model with advanced configuration."""
        print(f"🔄 Loading InternVL3.5-4B from {self.model_checkpoint}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_checkpoint, 
            trust_remote_code=True, 
            use_fast=False
        )
        print("✅ Tokenizer loaded")
        
        if use_quantization:
            # Advanced 4-bit quantization config
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            
            self.model = AutoModel.from_pretrained(
                self.model_checkpoint,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=False,  # Disabled for Kaggle compatibility
                trust_remote_code=True,
                device_map="auto"
            ).eval()
        else:
            self.model = AutoModel.from_pretrained(
                self.model_checkpoint,
                torch_dtype=self.data_type,
                low_cpu_mem_usage=True,
                use_flash_attn=False,
                trust_remote_code=True,
                device_map="auto"
            ).eval()
        
        print("✅ Model loaded!")
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            print(f"🧠 GPU Memory: {memory_allocated:.2f}GB")
    
    def build_transform(self, input_size=448):
        """Build image transformation pipeline."""
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """Find the closest aspect ratio from target ratios."""
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

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
        """Dynamic preprocessing with aspect ratio consideration."""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        target_aspect_ratio = self.find_closest_aspect_ratio(
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

    def load_image(self, image_file, input_size=448, max_num=12):
        """Load and preprocess image with dynamic preprocessing."""
        if isinstance(image_file, str) and image_file.startswith('http'):
            response = requests.get(image_file, timeout=10)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
            
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    def process_keyframe(self, image_path):
        """
        Process a single keyframe based on the task.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Generated response
        """
        try:
            # Load and preprocess image with dynamic preprocessing
            pixel_values = self.load_image(image_path, max_num=12)
            pixel_values = pixel_values.to(self.data_type).to(self.device)
            
            # For image_captioning task, apply banner/logo removal if needed
            if self.task == "image_captioning":
                # Note: Banner/logo removal can be added here if needed
                # For now, we rely on the improved preprocessing and model capabilities
                pass
            
            # Generate response using chat interface
            generation_config = dict(
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            question = f"<image>\n{self.prompt}"
            response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
            
            return response.strip()
            
        except Exception as e:
            print(f"Error processing keyframe {image_path}: {str(e)}")
            return "Image could not be processed"
    
    def process_batch(self, lesson_dir, output_dir):
        """
        Process all videos in a lesson directory.
        
        Args:
            lesson_dir (str): Directory containing video folders (e.g., "database/keyframes/L01")
            output_dir (str): Directory to save results (e.g., "database/caption" or "database/news_anchor")
        """
        os.makedirs(output_dir, exist_ok=True)
        lesson_name = os.path.basename(lesson_dir)
        
        output_lesson_dir = os.path.join(output_dir, lesson_name)
        os.makedirs(output_lesson_dir, exist_ok=True)

        videos = sorted(glob.glob(os.path.join(lesson_dir, "V*")))
        
        for video_dir in videos:
            video_name = os.path.basename(video_dir)
            
            keyframes = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
            
            if not keyframes:
                print(f"Warning: No keyframes found in {video_dir}")
                continue
            
            video_results = []
            for keyframe_path in tqdm(keyframes, desc=f"Processing {lesson_name}/{video_name} ({self.task})"):
                keyframe_name = os.path.basename(keyframe_path)
                result = self.process_keyframe(keyframe_path)
                
                if self.task == "image_captioning":
                    video_results.append({
                        "keyframe": keyframe_name, 
                        "caption": result
                    })
                elif self.task == "news_anchor_classification":
                    video_results.append({
                        "keyframe": keyframe_name, 
                        "prediction": result
                    })
            
            output_file = os.path.join(output_lesson_dir, f"{video_name}_{self.task}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(video_results, f, indent=2, ensure_ascii=False)
            
            print(f"Saved results to: {output_file}")
