import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from app.config import *
from app.init_app import load_database
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

# Khởi động ứng dụng
database = {}
with app.app_context():
    database = load_database()
    
    
# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/data/keyframes/<path:keyframe_name>')
def get_keyframe(keyframe_name):
    """Serve any file from the data/keyframes directory"""
    try:
        # Define paths
        keyframes_path = os.path.abspath(os.path.join(DATA_FOLDER, 'keyframes'))
        
        # Clean up the filename
        clean_filename = os.path.basename(keyframe_name).lstrip('/')
        
        # List all files in the keyframes directory
        try:
            all_files = os.listdir(keyframes_path)
        except Exception as e:
            return f"Error listing directory: {e}", 500
        
        # Try exact match first
        if clean_filename in all_files:
            file_path = os.path.join(keyframes_path, clean_filename)
            return send_from_directory(keyframes_path, clean_filename)
        
        # Try case-insensitive match
        filename_lower = clean_filename.lower()
        for file in all_files:
            if file.lower() == filename_lower:
                return send_from_directory(keyframes_path, file)
        
        # Try to find similar files (same base name, different extension)
        base_name = os.path.splitext(clean_filename)[0]
        similar_files = [f for f in all_files if os.path.splitext(f)[0] == base_name]
        if similar_files:
            return send_from_directory(keyframes_path, similar_files[0])
        
        # If we get here, log available files and return 404
        
        return "File not found", 404
        
    except Exception as e:
        return f"Internal server error: {str(e)}", 500


@app.route('/api/search', methods=['GET'])
def search():
    data = request.args

    try:
        query = data.get('query')
        models = data.get('models').split(',')
        top_k = int(data.get('top_k', 100))

        # Thực hiện tìm kiếm
        try:
            list_paths = {}
            for model in models:    
                faiss_handler = database[f'{model}']
                _, _, paths = faiss_handler.text_search(query=query, top_k=top_k)
                list_paths[model] = paths
            
            # Rerank
            paths, scores = rrf(list_paths, k_rrf=60)
            
            # Tạo response
            response_data = {
                'paths': [r.replace('data/', '', 1) for r in paths],
                'scores': [r for r in scores],
                'filenames': [os.path.basename(r) for r in paths]
            }
            
            response = jsonify(response_data)
            return response
            
        except Exception as e:
            error_msg = f'Error during search with {models}: {str(e)}'
            return jsonify({'error': error_msg}), 500
            
    except Exception as e:
        error_msg = f'Unexpected error during search: {str(e)}'
        return jsonify({
            'error': 'An error occurred during search',
            'details': str(e)
        }), 500
        

@app.route('/api/models', methods=['GET'])
def list_models():
    return jsonify({
        'models': list(EMBEDDING_MODELS.keys())
    })
    
@app.route('/api/objects', methods=['GET'])
def list_objects():
    return jsonify({
        'objects': OBJECTS
    })