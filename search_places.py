import os
import json
import glob

# ===== CONFIGURATION =====
# User can modify these parameters
PLACES_RESULTS_DIR = os.path.join("database", "places")
OUTPUT_FILE = "search_results.json"

# Place classes to search for - User can modify this list
TARGET_CLASSES = [
    "television_studio",
    "television_room"
]

def search_target_classes():
    """
    Search for keyframes matching target classes
    
    Returns:
        List of results with keyframe and prediction
    """
    results = []
    
    # Find all JSON result files
    json_pattern = os.path.join(PLACES_RESULTS_DIR, "**", "*_places.json")
    json_files = glob.glob(json_pattern, recursive=True)
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Search through all keyframes
        for item in data:
            keyframe = item.get('keyframe', '')
            prediction = item.get('prediction', '')
            
            # Check if prediction matches target classes (exact match)
            match_found = prediction in TARGET_CLASSES
            
            if match_found:
                results.append({
                    'keyframe': keyframe,
                    'prediction': prediction
                })
    
    return results

def main():
    # Search for target classes
    results = search_target_classes()
    
    # Count by prediction type
    prediction_counts = {}
    for result in results:
        pred = result['prediction']
        prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
    
    # Save results to JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main() 