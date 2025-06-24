import os
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from app.translate import translate_text
from app.database import Database
from app.rerank import rrf

# Khởi tạo Flask app
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Cấu hình CORS
cors = CORS()
cors.init_app(app, resources={
    r"/*": {
        "origins": ["http://localhost:5000", "http://127.0.0.1:5000"],
        "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
        "allow_headers": ["*"],
        "expose_headers": ["Content-Disposition"],
        "supports_credentials": False,
        "max_age": 600
    }
})

# Khởi tạo database
database = Database()
    

# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/data/keyframes/<path:keyframe_name>')
def get_keyframe(keyframe_name):
    """
    Input:
        - keyframe_name (string): Keyframe filename (e.g., "image_001.jpg")
    
    Output:
        - Success: Keyframe image file
        - Error: "File not found" (404)
    """
    keyframes_path = database.keyframes_path
    keyframe_name = os.path.basename(keyframe_name)
    
    file_path = os.path.join(keyframes_path, keyframe_name)
    if os.path.isfile(file_path):
        return send_from_directory(keyframes_path, keyframe_name)
    
    return "File not found", 404


@app.route('/api/search', methods=['GET'])
def search():
    """
    Input (URL Parameters):
        - query (string, optional): Text description of image to search (e.g., "a person riding a bicycle")
        - ocr_text (string, optional): OCR text to search in images (e.g., "STOP")
        - models (JSON array string, optional): List of models to use for search (e.g., '["CLIP ViT-B/32", "OpenCLIP ViT-B-32 laion2b_s34b_b79k"]')
        - objects (JSON array string, optional): List of objects to filter by (e.g., '["car", "person"]')
        - topK (int): Maximum number of results to return (e.g., 100)
    
    Output (JSON):
        {
            "paths": ["keyframes/image_001.jpg", "keyframes/image_002.jpg"],
            "scores": [0.95, 0.87],
            "filenames": ["image_001.jpg", "image_002.jpg"]
        }
    """
 
    query = translate_text(request.args.get('query', ''))
    ocr = request.args.get('ocr_text', '')
    models = request.args.get('models', '[]')
    objects = request.args.get('objects', '[]')
    topK = int(request.args.get('topK'))
    
    models = json.loads(models) if models else []
    objects = json.loads(objects) if objects else []

    list_paths = {}
    for model in models:    
        faiss_handler = database.embedding_models[f'{model}']
        _, _, paths = faiss_handler.text_search(query=query, top_k=topK)
        list_paths[model] = paths
    
    # TODO: Implement OCR text search logic here
    # TODO: Implement object filtering logic here

    paths, scores = rrf(list_paths, k_rrf=60)
    
    response_data = {
        'paths': [r.replace(database.database_path, '', 1) for r in paths],
        'scores': [r for r in scores],
        'filenames': [os.path.basename(r) for r in paths]
    }
    
    response = jsonify(response_data)
    return response
        

@app.route('/api/models', methods=['GET'])
def list_models():
    """
    Input: None
    
    Output (JSON):
        {
            "models": ["clip", "openclip", ...]
        }
    """
    return jsonify({
        'models': list(database.embedding_models.keys())
    })
    
@app.route('/api/objects', methods=['GET'])
def list_objects():
    """
    Input: None
    
    Output (JSON):
        {
            "objects": ["person", "car", "bicycle", ...]
        }
    """
    return jsonify({
        'objects': database.objects
    })