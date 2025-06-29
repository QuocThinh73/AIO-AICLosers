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
OUTPUT_DIR = "database/places"  # Save results to database/places

# Path to keyframes directory
KEYFRAMES_BASE_PATH = "database/keyframes"

# ===== PLACES365 CLASS NAMES =====
PLACES365_CLASSES = [
    "airfield", "airplane_cabin", "airport_terminal", "alcove", "alley", "amphitheater", "amusement_arcade", 
    "amusement_park", "apartment_building/outdoor", "aquarium", "aqueduct", "arcade", "arch", 
    "archaelogical_excavation", "archive", "arena/hockey", "arena/performance", "arena/rodeo", "army_base", 
    "art_gallery", "art_school", "art_studio", "artists_loft", "assembly_line", "athletic_field/outdoor", 
    "atrium/public", "attic", "auditorium", "auto_factory", "auto_showroom", "badlands", "bakery/shop", 
    "balcony/exterior", "balcony/interior", "ball_pit", "ballroom", "bamboo_forest", "bank_vault", 
    "banquet_hall", "bar", "barn", "barndoor", "baseball_field", "basement", "basketball_court/indoor", 
    "bathroom", "bazaar/indoor", "bazaar/outdoor", "beach", "beach_house", "beauty_salon", "bedchamber", 
    "bedroom", "beer_garden", "beer_hall", "berth", "biology_laboratory", "boardwalk", "boat_deck", 
    "boathouse", "bookstore", "booth/indoor", "botanical_garden", "bow_window/indoor", "bowling_alley", 
    "boxing_ring", "bridge", "building_facade", "bullring", "burial_chamber", "bus_interior", 
    "bus_station/indoor", "butchers_shop", "butte", "cabin/outdoor", "cafeteria", "campsite", "campus", 
    "canal/natural", "canal/urban", "candy_store", "canyon", "car_interior", "carrousel", "castle", 
    "catacomb", "cemetery", "chalet", "chemistry_lab", "childs_room", "church/indoor", "church/outdoor", 
    "classroom", "clean_room", "cliff", "closet", "clothing_store", "coast", "cockpit", "coffee_shop", 
    "computer_room", "conference_center", "conference_room", "construction_site", "corn_field", "corral", 
    "corridor", "cottage", "courthouse", "courtyard", "creek", "crevasse", "crosswalk", "dam", 
    "delicatessen", "department_store", "desert/sand", "desert/vegetation", "desert_road", "diner/outdoor", 
    "dining_hall", "dining_room", "discotheque", "doorway/outdoor", "dorm_room", "downtown", 
    "dressing_room", "driveway", "drugstore", "elevator/door", "elevator_lobby", "elevator_shaft", 
    "embassy", "engine_room", "entrance_hall", "escalator/indoor", "excavation", "fabric_store", "farm", 
    "fastfood_restaurant", "field/cultivated", "field/wild", "field_road", "fire_escape", "fire_station", 
    "fishpond", "flea_market/indoor", "florist_shop/indoor", "food_court", "football_field", 
    "forest/broadleaf", "forest_path", "forest_road", "formal_garden", "fountain", "galley", 
    "garage/indoor", "garage/outdoor", "gas_station", "gazebo/exterior", "general_store/indoor", 
    "general_store/outdoor", "gift_shop", "glacier", "golf_course", "greenhouse/indoor", 
    "greenhouse/outdoor", "grotto", "gymnasium/indoor", "hangar/indoor", "hangar/outdoor", "harbor", 
    "hardware_store", "hayfield", "heliport", "highway", "home_office", "home_theater", "hospital", 
    "hospital_room", "hot_spring", "hotel/outdoor", "hotel_room", "house", "hunting_lodge/outdoor", 
    "ice_cream_parlor", "ice_floe", "ice_shelf", "ice_skating_rink/indoor", "ice_skating_rink/outdoor", 
    "iceberg", "igloo", "industrial_area", "inn/outdoor", "islet", "jacuzzi/indoor", "jail_cell", 
    "japanese_garden", "jewelry_shop", "junkyard", "kasbah", "kennel/outdoor", "kindergarden_classroom", 
    "kitchen", "lagoon", "lake/natural", "landfill", "landing_deck", "laundromat", "lawn", "lecture_room", 
    "legislative_chamber", "library/indoor", "library/outdoor", "lighthouse", "living_room", 
    "loading_dock", "lobby", "lock_chamber", "locker_room", "mansion", "manufactured_home", 
    "market/indoor", "market/outdoor", "marsh", "martial_arts_gym", "mausoleum", "medina", "mezzanine", 
    "moat/water", "mosque/outdoor", "motel", "mountain", "mountain_path", "mountain_snowy", 
    "movie_theater/indoor", "museum/indoor", "museum/outdoor", "music_studio", "natural_history_museum", 
    "nursery", "nursing_home", "oast_house", "ocean", "office", "office_building", "office_cubicles", 
    "oilrig", "operating_room", "orchard", "orchestra_pit", "pagoda", "palace", "pantry", "park", 
    "parking_garage/indoor", "parking_garage/outdoor", "parking_lot", "pasture", "patio", "pavilion", 
    "pet_shop", "pharmacy", "phone_booth", "physics_laboratory", "picnic_area", "pier", "pizzeria", 
    "playground", "playroom", "plaza", "pond", "porch", "promenade", "pub/indoor", "racecourse", 
    "raceway", "raft", "railroad_track", "rainforest", "reception", "recreation_room", "repair_shop", 
    "residential_neighborhood", "restaurant", "restaurant_kitchen", "restaurant_patio", "rice_paddy", 
    "river", "rock_arch", "roof_garden", "rope_bridge", "ruin", "runway", "sandbox", "sauna", 
    "schoolhouse", "science_museum", "server_room", "shed", "shoe_shop", "shopfront", 
    "shopping_mall/indoor", "shower", "ski_resort", "ski_slope", "sky", "skyscraper", "slum", 
    "snowfield", "soccer_field", "stable", "stadium/baseball", "stadium/football", "stadium/soccer", 
    "stage/indoor", "stage/outdoor", "staircase", "storage_room", "street", "subway_station/platform", 
    "supermarket", "sushi_bar", "swamp", "swimming_hole", "swimming_pool/indoor", "swimming_pool/outdoor", 
    "synagogue/outdoor", "television_room", "television_studio", "temple/asia", "throne_room", 
    "ticket_booth", "topiary_garden", "tower", "toyshop", "train_interior", "train_station/platform", 
    "tree_farm", "tree_house", "trench", "tundra", "underwater/ocean_deep", "utility_room", "valley", 
    "vegetable_garden", "veterinarians_office", "viaduct", "village", "vineyard", "volcano", 
    "volleyball_court/outdoor", "waiting_room", "water_park", "water_tower", "waterfall", 
    "watering_hole", "wave", "wet_bar", "wheat_field", "wind_farm", "windmill", "yard", "youth_hostel", 
    "zen_garden"
]

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
