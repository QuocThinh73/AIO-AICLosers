import os
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
import json
import traceback

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

def classify_keyframe(keyframe_path, model, processor):
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
    output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    answer = processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
    return 1 if "YES" in answer else 0

def process_video(video_path, news_anchor_path, model, processor):
    results = []
    for keyframe_file in sorted(os.listdir(video_path)):
        keyframe_path = os.path.join(video_path, keyframe_file)
        prediction = classify_keyframe(keyframe_path, model, processor)
        results.append({
            "keyframe_path": keyframe_path,
            "prediction": prediction
        })
        
    with open(news_anchor_path, "w") as f:
        json.dump(results, f)
        
        
def detect_news_anchor(input_keyframe_dir, output_news_anchor_dir, mode, lesson_name=None):
    os.makedirs(output_news_anchor_dir, exist_ok=True)
    
    model, processor = load_model()
    
    if mode == "lesson":
        os.makedirs(os.path.join(output_news_anchor_dir, lesson_name), exist_ok=True)
        for video_folder in sorted(os.listdir(os.path.join(input_keyframe_dir, lesson_name))):
            video_path = os.path.join(input_keyframe_dir, lesson_name, video_folder)
            news_anchor_path = os.path.join(output_news_anchor_dir, lesson_name, f"{lesson_name}_{video_folder}_news_anchor.json")
            process_video(video_path, news_anchor_path, model, processor)
            
    elif mode == "all":
        for lesson_folder in sorted(os.listdir(input_keyframe_dir)):
            os.makedirs(os.path.join(output_news_anchor_dir, lesson_folder), exist_ok=True)
            for video_folder in sorted(os.listdir(os.path.join(input_keyframe_dir, lesson_folder))):
                video_path = os.path.join(input_keyframe_dir, lesson_folder, video_folder)
                news_anchor_path = os.path.join(output_news_anchor_dir, lesson_folder, f"{lesson_folder}_{video_folder}_news_anchor.json")
                process_video(video_path, news_anchor_path, model, processor)