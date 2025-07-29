import torch
import os
import json
import glob
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig


class InternVL3:
    def __init__(self, task, model_checkpoint="OpenGVLab/InternVL3-8B-hf", max_new_tokens=100, use_quantization=True):
        """
        Initialize the InternVL3 model for different tasks.
        
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
        
        # Set task-specific prompts
        self._setup_prompt()
        self._load_model(use_quantization)
    
    def _setup_prompt(self):
        """Setup task-specific prompts."""
        if self.task == "image_captioning":
            self.prompt = """
You are an image captioning assistant processing still frames from live broadcast news videos. 
Provide a concise but informative description that mentions the main subjects (people or objects), their actions, and the scene context. 
Ignore any on-screen graphics such as ticker text, news banners, program logos, watermarks, or other unrelated overlays.
"""
            
        elif self.task == "news_anchor_classification":
            self.prompt = """
Does the image show a news anchor **actively presenting** news in a professional **broadcast TV studio** (e.g. desk with news branding, lighting rigs, large studio screens, official newsroom setup)?

Only answer YES if:
– The person is clearly delivering news (e.g. reading script, facing camera).
– The environment includes professional TV studio features.

Otherwise, answer NO.

Do not be misled by graphics or text overlays. 
Only say YES if the person is a real news anchor in a live TV studio with full broadcasting setup. 
Ignore marketing displays, showroom backgrounds, or mock studio designs.

Answer with only YES or NO.
"""
    
    def _load_model(self, use_quantization=True):
        """Load the processor and model."""
        self.processor = AutoProcessor.from_pretrained(self.model_checkpoint)
        
        if use_quantization:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_checkpoint, 
                quantization_config=quantization_config, 
                torch_dtype=self.data_type
            ).eval().to(self.device)
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_checkpoint, 
                torch_dtype=self.data_type
            ).eval().to(self.device)
    
    def process_keyframe(self, image_path):
        """
        Process a single keyframe based on the task.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Generated response
        """
        # For image_captioning task, preprocess the image: resize and remove banner/logo
        if self.task == "image_captioning":
            import cv2
            import os
            from utils import delete_banner_and_logo
            
            # Define target size and regions to mask
            target_size = (1280, 720)
            logo_box = (1000, 50, 1300, 130)
            banner_box = (0, 660, 1280, 690)
            mask_boxes = [logo_box, banner_box]
            
            # Read and resize image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}")
                return "Image could not be processed"
            
            # Resize image to target size
            img_resized = cv2.resize(img, target_size)
            
            # Mask out banner and logo regions
            masked_img = delete_banner_and_logo(img_resized.copy(), mask_boxes)
            
            # Convert numpy array to base64 string
            import base64
            from io import BytesIO
            from PIL import Image
            
            # Convert BGR to RGB (CV2 uses BGR, PIL uses RGB)
            masked_img_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(masked_img_rgb)
            
            # Save to buffer and convert to base64
            buffered = BytesIO()
            pil_img.save(buffered, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # Create data URL for the image
            image_path = f"data:image/jpeg;base64,{img_b64}"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": self.prompt}
                ]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt"
        ).to(self.model.device, dtype=self.data_type)
        
        output = self.model.generate(
            **inputs, 
            max_new_tokens=self.max_new_tokens, 
            do_sample=False
        )
        
        answer = self.processor.decode(
            output[0, inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        return answer
    
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
