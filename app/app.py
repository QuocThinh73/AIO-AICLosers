import os
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from app.config import *
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
    try:
        keyframes_path = os.path.abspath(os.path.join(DATA_FOLDER, 'keyframes'))
        keyframe_name = os.path.basename(keyframe_name)
        
        file_path = os.path.join(keyframes_path, keyframe_name)
        if os.path.isfile(file_path):
            return send_from_directory(keyframes_path, keyframe_name)
        
        return "File not found", 404
        
    except Exception as e:
        return f"Internal server error: {str(e)}", 500


@app.route('/api/search', methods=['GET'])
def search():
    data = request.args

    try:
        query = data.get('query')            
        models = json.loads(data.get('models'))
        topK = int(data.get('topK', 100))

        # Thực hiện tìm kiếm
        try:
            list_paths = {}
            for model in models:    
                faiss_handler = database.embedding_models[f'{model}']
                _, _, paths = faiss_handler.text_search(query=query, top_k=topK)
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