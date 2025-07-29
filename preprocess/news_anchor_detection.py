import os
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
import json
import traceback
import time
import concurrent.futures
from threading import Timer

PROMPT = """
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

def load_model():
    try:
        print("Loading InternVL3 model for news anchor detection...")
        model = AutoModelForImageTextToText.from_pretrained("OpenGVLab/InternVL3-2B-hf", torch_dtype=torch.float32, device_map="auto")
        processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL3-2B-hf")
        print("Model loaded successfully!")
        return model, processor
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print(traceback.format_exc())
        raise

def classify_keyframe_with_timeout(keyframe_path, model, processor, timeout_seconds=30):
    """Classify a keyframe with a timeout to prevent hanging"""
    result = None
    exception = None
    
    # Define the worker function
    def worker():
        nonlocal result, exception
        try:
            print(f"Processing keyframe: {os.path.basename(keyframe_path)}")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": keyframe_path},
                        {"type": "text", "text": PROMPT}
                    ]
                }
            ]
            
            inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.float32)
            print(f"  - Input processed, generating output...")
            output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            print(f"  - Output generated, decoding...")
            answer = processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
            print(f"  - Result for {os.path.basename(keyframe_path)}: {answer}")
            result = 1 if "YES" in answer else 0
        except Exception as e:
            exception = e
            print(f"Error classifying keyframe {os.path.basename(keyframe_path)}: {str(e)}")
    
    # Run with timeout
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker)
        try:
            future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            print(f"WARNING: Classification timed out after {timeout_seconds}s for {os.path.basename(keyframe_path)}")
            return 0  # Default to 'NO' for timeouts
    
    if exception:
        print(f"ERROR: Exception while processing {os.path.basename(keyframe_path)}: {exception}")
        return 0  # Default to 'NO' for errors
    
    return result

def process_video(video_path, news_anchor_path, model, processor):
    print(f"\nProcessing video folder: {os.path.basename(video_path)}")
    start_time = time.time()
    
    # Chỉ xử lý các file hình ảnh
    image_extensions = [".jpg", ".jpeg", ".png"]
    keyframe_files = [f for f in sorted(os.listdir(video_path)) 
                    if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    results = []
    total_files = len(keyframe_files)
    
    print(f"Found {total_files} keyframes to process")
    
    for i, keyframe_file in enumerate(keyframe_files):
        keyframe_path = os.path.join(video_path, keyframe_file)
        print(f"[{i+1}/{total_files}] Processing {keyframe_file}...")
        
        # Process with timeout to prevent hanging
        prediction = classify_keyframe_with_timeout(keyframe_path, model, processor)
        
        results.append({
            "keyframe_path": keyframe_path,
            "prediction": prediction
        })
        
        # Save intermediate results every 5 frames
        if (i+1) % 5 == 0 or i == total_files-1:
            print(f"Saving intermediate results to {news_anchor_path}")
            os.makedirs(os.path.dirname(news_anchor_path), exist_ok=True)
            with open(news_anchor_path, "w") as f:
                json.dump(results, f, indent=2)
    
    elapsed_time = time.time() - start_time
    print(f"Completed processing {total_files} keyframes in {elapsed_time:.2f} seconds")
    
    # Final save
    with open(news_anchor_path, "w") as f:
        json.dump(results, f, indent=2)
        
        
def detect_news_anchor(input_keyframe_dir, output_news_anchor_dir, mode, lesson_name=None):
    os.makedirs(output_news_anchor_dir, exist_ok=True)
    
    model, processor = load_model()
    
    if mode == "lesson":
        os.makedirs(os.path.join(output_news_anchor_dir, lesson_name), exist_ok=True)
        for video_folder in sorted(os.listdir(os.path.join(input_keyframe_dir, lesson_name))):
            video_path = os.path.join(input_keyframe_dir, lesson_name, video_folder)
            news_anchor_path = os.path.join(output_news_anchor_dir, lesson_name, f"{lesson_name}_{video_folder}_news_anchor.json")
            process_video(video_path, news_anchor_path, model, processor)
            print(f"Completed processing {video_folder}")
            
    elif mode == "all":
        for lesson_folder in sorted(os.listdir(input_keyframe_dir)):
            os.makedirs(os.path.join(output_news_anchor_dir, lesson_folder), exist_ok=True)
            for video_folder in sorted(os.listdir(os.path.join(input_keyframe_dir, lesson_folder))):
                video_path = os.path.join(input_keyframe_dir, lesson_folder, video_folder)
                news_anchor_path = os.path.join(output_news_anchor_dir, lesson_folder, f"{lesson_folder}_{video_folder}_news_anchor.json")
                process_video(video_path, news_anchor_path, model, processor)
                print(f"Completed processing {lesson_folder}_{video_folder}")