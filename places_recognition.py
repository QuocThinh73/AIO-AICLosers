import os
import numpy as np
import torch
import birder
import json
import glob
from tqdm import tqdm

# ===== CONFIGURATION PARAMETERS =====
# User can modify these parameters
BATCH_L = "L01"  # Batch to process (can change to L02, L03, ...)
MODEL_NAME = "rope_vit_reg4_b14_capi-places365"
OUTPUT_DIR = os.path.join("database", "places")  # Save results to database/places

# Path to keyframes directory
KEYFRAMES_BASE_PATH = os.path.join("database", "keyframes")

# ===== PLACES365 CLASS NAMES =====
def load_places365_classes():
    """Load Places365 class names from external JSON file"""
    classes_file = os.path.join("database", "places365_classes.json")
    with open(classes_file, 'r', encoding='utf-8') as f:
        return json.load(f)

# Global variable to store classes (loaded once)
PLACES365_CLASSES = None

def load_model():
    """Load pre-trained birder model"""
    # Load pre-trained birder model
    (net, model_info) = birder.load_pretrained_model(MODEL_NAME, inference=True)
    
    # Get the image size the model was trained on
    size = birder.get_size_from_signature(model_info.signature)
    
    # Create an inference transform
    transform = birder.classification_transform(size, model_info.rgb_stats)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    
    return net, transform, device

def get_keyframes_for_batch(batch_l: str):
    """
    Get all keyframes for the specified batch L
    
    Returns:
        Dict with video name as key and list of keyframe paths as value
    """
    batch_keyframes = {}
    batch_path = os.path.join(KEYFRAMES_BASE_PATH, batch_l)
    
    # Scan video directories in batch
    for v_item in os.listdir(batch_path):
        v_path = os.path.join(batch_path, v_item)
        if os.path.isdir(v_path) and v_item.startswith('V'):
            # Get all keyframes in video directory (format: L01_V001_*.jpg)
            pattern = os.path.join(v_path, "*.jpg")
            keyframes = sorted(glob.glob(pattern))
            if keyframes:
                video_name = f"{batch_l}_{v_item}"  # Example: L01_V001
                batch_keyframes[video_name] = keyframes
    
    return batch_keyframes



def predict_image(net, transform, image_path: str):
    """Predict and extract embedding from image"""
    from birder.inference.classification import infer_image
    
    # Perform inference
    (predictions, embedding) = infer_image(net, image_path, transform, return_embedding=True)
    
    # Convert prediction scores to class name
    pred_class_idx = np.argmax(predictions)
    pred_class_name = PLACES365_CLASSES[pred_class_idx]
    
    return pred_class_name, embedding

def process_keyframes_batch(net, transform, keyframe_paths):
    """Process a batch of keyframes"""
    results = {
        'predictions': [],
        'embeddings': [],
        'keyframe_paths': keyframe_paths
    }
    
    for keyframe_path in tqdm(keyframe_paths, desc="Processing keyframes"):
        pred_class, emb = predict_image(net, transform, keyframe_path)
        results['predictions'].append(pred_class)
        results['embeddings'].append(emb.tolist())
    
    return results

def save_video_results(video_name, results, output_base, batch_l):
    """Save results for one video in a single file"""
    # Create directory for batch
    batch_output_dir = os.path.join(output_base, batch_l)
    os.makedirs(batch_output_dir, exist_ok=True)
    
    # Prepare keyframes data - only 3 fields: keyframe, prediction, embedding
    keyframes_data = []
    for i, keyframe_path in enumerate(results['keyframe_paths']):
        keyframe_name = os.path.basename(keyframe_path)
        keyframe_data = {
            'keyframe': keyframe_name,
            'prediction': results['predictions'][i],
            'embedding': results['embeddings'][i]
        }
        keyframes_data.append(keyframe_data)
    
    # Save results in single file per video
    results_file = os.path.join(batch_output_dir, f"{video_name}_places.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(keyframes_data, f, indent=2, ensure_ascii=False)

def main():
    global PLACES365_CLASSES
    
    # Load Places365 classes
    PLACES365_CLASSES = load_places365_classes()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    net, transform, device = load_model()
    
    # Load keyframes for specified batch
    batch_keyframes = get_keyframes_for_batch(BATCH_L)
    
    total_keyframes = 0
    for video_name, keyframes in batch_keyframes.items():
        total_keyframes += len(keyframes)
    
    # Process each video
    for video_name, keyframes in batch_keyframes.items():        
        # Process keyframes with birder model
        results = process_keyframes_batch(net, transform, keyframes)
        
        # Save results (returns single file per video)
        save_video_results(video_name, results, OUTPUT_DIR, BATCH_L)

if __name__ == "__main__":
    main()
