import os
import cv2
import warnings
from flask import Flask, jsonify, render_template, send_from_directory

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*QuickGELU.*")
warnings.filterwarnings("ignore", message=".*timm.models.layers.*")
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

# Standard imports for Python package execution
from .database import Database
from .handlers.request_handler import parse_search_request
from .handlers.search_handler import perform_unified_search, format_search_response

# Initialize Flask application with static and template folders
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Initialize database connection and models
database = Database()
    

# Routes
@app.route('/')
def index():
    """
    Serve the main application page.
    
    Returns:
        HTML template for the search interface
    """
    return render_template('index.html')


@app.route('/data/keyframes/<path:keyframe_name>')
def get_keyframe(keyframe_name):
    """
    Serve keyframe images from the database.
    
    Args:
        keyframe_name (str): Can be in format "L01_V003_015190.jpg" or "L01/V003/L01_V003_015190.jpg"
    
    Returns:
        File: Keyframe image file if found
        str: "File not found" with 404 status if not found
    """
    
    # Handle nested path format: L01/V003/L01_V003_015190.jpg
    if '/' in keyframe_name:
        path_parts = keyframe_name.split('/')
        filename = path_parts[-1]  # Get the filename from the end
        
        # Attempt to determine folder path based on directory structure
        lesson_folder = path_parts[0]  # L01
        video_folder = path_parts[1] if len(path_parts) > 2 else path_parts[0]  # V003 or fallback
    else:
        # Traditional parsing: L01_V003_015190.jpg
        filename = keyframe_name
        parts = filename.split('_')
        if len(parts) >= 2:
            lesson_folder = parts[0]  # L01
            video_folder = parts[1]   # V003
        else:
            # Fallback if can't parse
            print(f"[WARNING] Invalid keyframe name format: {keyframe_name}")
            return "Invalid keyframe format", 400
    
    folder_path = os.path.join(database.keyframes_path, lesson_folder, video_folder)
    file_path = os.path.join(folder_path, filename)
    
    if os.path.isfile(file_path):
        return send_from_directory(folder_path, filename)
    else:
        print(f"[WARNING] Keyframe file not found: {file_path}")
    
    return "File not found", 404


@app.route('/data/videos/<path:video_name>')
def get_video(video_name):
    """
    Serve video files from the database.
    
    Args:
        video_name (str): Video filename in format "L01_V003.mp4"
    
    Returns:
        File: Video file if found
        str: "File not found" with 404 status if not found
    """
    
    # Parse video filename to extract lesson folder
    parts = video_name.split('_')
    lesson_folder = parts[0]  # L01
    folder_path = os.path.join(database.videos_path, lesson_folder)
    file_path = os.path.join(folder_path, video_name)
    
    if os.path.isfile(file_path):
        return send_from_directory(folder_path, video_name)
    
    return "File not found", 404


@app.route('/api/video-info/<path:keyframe_name>')
def get_video_info(keyframe_name):
    """
    Get video information and timestamp for a given keyframe.
    
    Args:
        keyframe_name (str): Keyframe filename in format "L01_V003_015190.jpg" 
                            or nested path format
    
    Returns:
        JSON: Video information including path, timestamp, frame number, and FPS
              or error message if video not found
    """

    try:
        # Handle different path formats and extract filename
        if '/' in keyframe_name:
            filename = keyframe_name.split('/')[-1]
        else:
            filename = keyframe_name
        
        # Parse keyframe filename: L01_V003_015190.jpg
        parts = filename.split('_')
        if len(parts) < 3:
            return jsonify({
                'error': f'Invalid keyframe filename format: {filename}'
            }), 400
            
        lesson = parts[0]  # L01
        video = parts[1]   # V003
        frame_str = parts[2].replace('.jpg', '')  # 015190
            
        frame_number = int(frame_str)
        
        # Build actual video path to read FPS
        video_name = f"{lesson}_{video}.mp4"
        actual_video_path = os.path.join(database.videos_path, lesson, video_name)
        
        if os.path.isfile(actual_video_path):
            # Read actual FPS from video file
            cap = cv2.VideoCapture(actual_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            # Calculate timestamp from frame number and actual FPS
            timestamp = frame_number / fps
            
            video_path = f"/data/videos/{video_name}"
            
            return jsonify({
                'video_path': video_path,
                'timestamp': timestamp,
                'frame_number': frame_number,
                'fps': fps
            })
        else:
            print(f"[WARNING] Video not found: {actual_video_path}")
            return jsonify({
                'error': 'Video not found',
                'keyframe': filename
            }), 404
    except Exception as e:
        print(f"[ERROR] Error processing video info: {str(e)}")
        return jsonify({
            'error': 'Error processing video information',
            'message': str(e)
        }), 500


@app.route('/api/video-keyframes/<path:keyframe_name>')
def get_video_keyframes(keyframe_name):
    """
    Get all keyframes for the same video as the given keyframe.
    
    Args:
        keyframe_name (str): Keyframe filename in format "L01_V003_015190.jpg"
    
    Returns:
        JSON: List of all keyframes for the same video, sorted by frame number
        str: "Directory not found" with 404 status if keyframes directory not found
    """
    
    # Parse keyframe filename: L01_V003_015190.jpg
    parts = keyframe_name.split('_')
    lesson = parts[0]  # L01
    video = parts[1]   # V003
    
    # Build keyframes directory path
    keyframes_dir = os.path.join(database.keyframes_path, lesson, video)
    
    if not os.path.isdir(keyframes_dir):
        return "Directory not found", 404
    
    # Get all keyframe files for this video
    keyframes = []
    for filename in os.listdir(keyframes_dir):
        if filename.endswith('.jpg') and filename.startswith(f"{lesson}_{video}_"):
            # Extract frame number for sorting
            frame_str = filename.split('_')[2].replace('.jpg', '')
            frame_number = int(frame_str)
            keyframes.append({
                'filename': filename,
                'frame_number': frame_number,
                'path': f"/data/keyframes/{filename}"
            })
    
    # Sort by frame number
    keyframes.sort(key=lambda x: x['frame_number'])
    
    return jsonify({
        'keyframes': keyframes,
        'current_keyframe': keyframe_name
    })


@app.route('/api/search', methods=['POST'])
def search():
    """
    Unified search endpoint for keyframes using text, images, OCR, and object filters.
    
    Args (POST FormData):
        file (file, optional): Uploaded image file for image search
        query (str, optional): Text description of image to search
        ocr_text (str, optional): OCR text to search in images  
        models (str): JSON array of models to use for search
        objects (str, optional): JSON array of objects to filter by
        topK (int): Maximum number of results to return
    
    Returns:
        JSON: Search results with paths, scores, and filenames
    """
    try:
        print("[DEBUG] Search request received")
        
        # Parse request data
        uploaded_image, search_params = parse_search_request()
        print(f"[DEBUG] Search params: {search_params}")
        print(f"[DEBUG] Has uploaded image: {uploaded_image is not None}")
        
        # Perform unified search
        paths, scores = perform_unified_search(uploaded_image, search_params, database)
        print(f"[DEBUG] Search returned {len(paths)} results")
        
        # Format and return response
        response_data = format_search_response(paths, scores, uploaded_image, search_params, database)
        print(f"[DEBUG] Response data keys: {list(response_data.keys())}")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"[ERROR] Search failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



@app.route('/api/models', methods=['GET'])
def list_models():
    """
    Get list of available embedding models.
    
    Returns:
        JSON: Dictionary containing list of available model names
    """
    return jsonify({
        'models': list(database.embedding_models.keys())
    })
    
@app.route('/api/objects', methods=['GET'])
def list_objects():
    """
    Get list of available object classes for filtering.
    
    Returns:
        JSON: Dictionary containing list of available object classes
    """
    return jsonify({
        'objects': database.objects
    })


if __name__ == '__main__':
    print("[INFO] Starting AIO-AIClosers Flask App")
    print("=" * 50)
    print(f"Database initialized with {len(database.get_embedding_models())} embedding models")
    print(f"GroundingDINO: {'OK' if database.grounding_dino and database.grounding_dino.model_loaded else 'FAIL'}")
    print(f"Object classes: {len(database.objects)}")
    print("\n[INFO] Starting Flask server...")
    print("   URL: http://localhost:5000")
    print("   Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=False  # Disable reloader to avoid double initialization
        )
    except KeyboardInterrupt:
        print("\n[INFO] App stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Error running app: {e}")
        import traceback
        traceback.print_exc()