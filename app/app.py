import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import torch
import sys
from models.clip import CLIP
from models.openclip import OpenCLIP
from faiss_index import FaissIndex

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
        "supports_credentials": False,  # Tắt supports_credentials
        "max_age": 600
    }
})

# Cấu hình
app.config.update(
    UPLOAD_FOLDER='app/static/images',
    DATA_FOLDER='data',
    DATABASE_FOLDER='database',
    MODEL_DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
)

# Tạo thư mục nếu chưa tồn tại
for folder in [app.config['UPLOAD_FOLDER'], 
               os.path.join(app.config['DATA_FOLDER'], 'keyframes'),
               app.config['DATABASE_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Khởi tạo models và FAISS indexes
models = {}
faiss_handlers = {}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_models_and_indexes():
    """Tải các model và FAISS indexes từ thư mục database"""
    global models, faiss_handlers
    
    # In thông tin thư mục database (sử dụng encoding phù hợp cho Windows)
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    
    database_path = os.path.abspath(app.config['DATABASE_FOLDER'])
    print("\n" + "="*50)
    print(f"Đang tải chỉ mục từ thư mục: {database_path}")
    
    # Kiểm tra xem thư mục tồn tại không
    if not os.path.exists(database_path):
        print(f"LỖI: Thư mục {database_path} không tồn tại")
        return
    
    # Khởi tạo các models
    try:
        print("\nĐang tải CLIP model...")
        try:
            clip_model = CLIP(device=device)
            models['clip'] = clip_model
            print("Đã tải xong CLIP model")
        except Exception as e:
            print(f"Không thể tải CLIP model: {str(e)}")
            
        print("\nĐang tải OpenCLIP model...")
        try:
            openclip_model = OpenCLIP(backbone='ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
            models['openclip'] = openclip_model
            print("Đã tải xong OpenCLIP model")
        except Exception as e:
            print(f"Không thể tải OpenCLIP model: {str(e)}")
        
        # In danh sách các model đã tải thành công
        if models:
            print("\nCác model đã tải thành công:")
            for name in models.keys():
                print(f"- {name}")
        else:
            print("\nCảnh báo: Không có model nào được tải thành công")
            return
            
        # Tải các FAISS index tương ứng
        for model_name, model in models.items():
            index_path = os.path.join(database_path, f'{model_name}_faiss.bin')
            id_map_path = os.path.join(database_path, f'{model_name}_id2path.pkl')
            
            if not os.path.exists(index_path):
                print(f"Cảnh báo: Không tìm thấy file index cho model {model_name}: {index_path}")
                continue
                
            if not os.path.exists(id_map_path):
                print(f"Cảnh báo: Không tìm thấy file id map cho model {model_name}: {id_map_path}")
                continue
                
            try:
                print(f"\nĐang tải FAISS index cho {model_name}...")
                faiss_handler = FaissIndex(model=model)
                faiss_handler.load(index_path, id_map_path)
                faiss_handlers[model_name] = faiss_handler
                print(f"Đã tải xong FAISS index cho {model_name}")
            except Exception as e:
                print(f"Lỗi khi tải FAISS index cho {model_name}: {str(e)}")
    
    except Exception as e:
        import traceback
        print(f"Lỗi khi khởi tạo models: {str(e)}\n{traceback.format_exc()}")
    
    # In thông tin các model đã tải
    print("\n" + "="*50)
    print("TÓM TẮT CÁC MODEL ĐÃ TẢI:")
    if not faiss_handlers:
        print("KHÔNG CÓ MODEL NÀO ĐƯỢC TẢI THÀNH CÔNG")
    else:
        for model_name, handler in faiss_handlers.items():
            print(f"- {model_name}: Đã sẵn sàng")
    print("="*50 + "\n")

# Khởi tạo models và indexes khi khởi động ứng dụng
with app.app_context():
    load_models_and_indexes()

# Routes
@app.route('/')
def index():
    """Trang chủ - hiển thị giao diện tìm kiếm"""
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Phục vụ các file tĩnh"""
    return send_from_directory('static', filename)

@app.route('/data/<path:filename>')
def serve_data(filename):
    """Serve any file from the data/keyframes directory"""
    try:
        # Define paths
        keyframes_path = os.path.abspath(os.path.join(app.config['DATA_FOLDER'], 'keyframes'))
        
        # Log the incoming request
        app.logger.info(f"\n=== New Request ===")
        app.logger.info(f"Request URL: {request.url}")
        app.logger.info(f"Requested filename: {filename}")
        
        # Clean up the filename
        clean_filename = os.path.basename(filename).lstrip('/')
        app.logger.info(f"Cleaned filename: {clean_filename}")
        
        # Verify the keyframes directory exists
        if not os.path.exists(keyframes_path):
            app.logger.error(f"Keyframes directory not found: {keyframes_path}")
            return f"Keyframes directory not found: {keyframes_path}", 500
        
        # List all files in the keyframes directory
        try:
            all_files = os.listdir(keyframes_path)
            app.logger.info(f"Found {len(all_files)} files in keyframes directory")
        except Exception as e:
            app.logger.error(f"Error listing directory {keyframes_path}: {e}")
            return f"Error listing directory: {e}", 500
        
        # Try exact match first
        if clean_filename in all_files:
            file_path = os.path.join(keyframes_path, clean_filename)
            app.logger.info(f"✓ Found exact match: {file_path}")
            return send_from_directory(keyframes_path, clean_filename)
        
        # Try case-insensitive match
        filename_lower = clean_filename.lower()
        for file in all_files:
            if file.lower() == filename_lower:
                app.logger.info(f"✓ Found case-insensitive match: {file}")
                return send_from_directory(keyframes_path, file)
        
        # Try to find similar files (same base name, different extension)
        base_name = os.path.splitext(clean_filename)[0]
        similar_files = [f for f in all_files if os.path.splitext(f)[0] == base_name]
        if similar_files:
            app.logger.info(f"✓ Found similar file: {similar_files[0]}")
            return send_from_directory(keyframes_path, similar_files[0])
        
        # If we get here, log available files and return 404
        app.logger.error("\n=== File Not Found ===")
        app.logger.error(f"Requested file: {clean_filename}")
        app.logger.error(f"Available files (first 20): {all_files[:20]}")
        
        return "File not found", 404
        
    except Exception as e:
        app.logger.error(f"Unexpected error in serve_data: {str(e)}", exc_info=True)
        return f"Internal server error: {str(e)}", 500


@app.route('/api/search', methods=['GET', 'POST', 'OPTIONS'])
def search():
    if request.method == 'OPTIONS':
        # Xử lý preflight request
        response = jsonify({'status': 'preflight'})
        origin = request.headers.get('Origin', '*')
        if origin in ['http://localhost:5000', 'http://127.0.0.1:5000']:
            response.headers.add('Access-Control-Allow-Origin', origin)
        else:
            response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5000')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        response.headers.add('Access-Control-Max-Age', '600')
        return response
    
    # Xử lý dữ liệu đầu vào
    if request.method == 'GET':
        # Lấy tham số từ query string
        data = request.args
    else:  # POST
        # Kiểm tra content-type
        if not request.is_json:
            app.logger.error('Invalid content-type. Expected application/json')
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        data = request.get_json()
        if not data:
            app.logger.error('No data received')
            return jsonify({'error': 'No data received'}), 400
    
    try:
        app.logger.info('Received search request')
        
        # Lấy tham số từ dữ liệu đầu vào
        query = data.get('query')
        model_name = data.get('model_name', 'openclip')  # Mặc định là openclip
        top_k = min(int(data.get('top_k', 12)), 50)  # Giới hạn tối đa 50 kết quả

        if not query:
            app.logger.error('No query provided')
            return jsonify({'error': 'Query is required'}), 400

        app.logger.info(f'Search params - query: {query}, model: {model_name}, top_k: {top_k}')

        # Kiểm tra model có tồn tại không
        if model_name not in faiss_handlers:
            available_models = list(faiss_handlers.keys())
            error_msg = f'Model {model_name} not found or not loaded. Available models: {available_models}'
            app.logger.error(error_msg)
            return jsonify({
                'error': error_msg,
                'available_models': available_models
            }), 400

        # Thực hiện tìm kiếm
        try:
            app.logger.info(f'Searching with {model_name}...')
            faiss_handler = faiss_handlers[model_name]
            
            # Gọi phương thức text_search từ FaissIndex
            scores, indices, paths = faiss_handler.text_search(query, top_k=top_k)
            
            app.logger.info(f'Search completed. Found {len(paths)} results')
            
            # Tạo danh sách kết quả
            results = []
            for score, idx, path in zip(scores, indices, paths):
                results.append({
                    'path': path,
                    'score': float(score)
                })
            
            # Sắp xếp kết quả theo score (từ cao xuống thấp)
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # Tạo response
            response_data = {
                'query': query,
                'model': model_name,
                'paths': [r['path'].replace('data/', '', 1) for r in results],
                'scores': [r['score'] for r in results],
                'filenames': [os.path.basename(r['path']) for r in results]
            }
            
            response = jsonify(response_data)
            origin = request.headers.get('Origin', '*')
            if origin in ['http://localhost:5000', 'http://127.0.0.1:5000']:
                response.headers.add('Access-Control-Allow-Origin', origin)
            else:
                response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5000')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            response.headers.add('Vary', 'Origin')
            return response
            
        except Exception as e:
            error_msg = f'Error during search with {model_name}: {str(e)}'
            app.logger.error(error_msg, exc_info=True)
            return jsonify({'error': error_msg, 'details': str(e)}), 500
            
    except Exception as e:
        error_msg = f'Unexpected error during search: {str(e)}'
        app.logger.error(error_msg, exc_info=True)
        return jsonify({
            'error': 'An error occurred during search',
            'details': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Kiểm tra trạng thái của ứng dụng"""
    status = {
        "status": "ok",
        "service": "image-search-api",
        "models_loaded": list(models.keys()),
        "faiss_indexes_loaded": list(faiss_indexes.keys()),
        "device": app.config['MODEL_DEVICE']
    }
    
    # Kiểm tra số lượng ảnh trong id_maps
    for model_name, id_map in id_maps.items():
        status[f"{model_name}_images"] = len(id_map)
    
    return jsonify(status)

@app.route('/api/models', methods=['GET'])
def list_models():
    """Liệt kê các model đã tải"""
    try:
        # Lấy danh sách các model đã được tải
        available_models = list(faiss_handlers.keys())
        
        # Tạo thông tin chi tiết về từng model
        models_info = {}
        for model_name in available_models:
            models_info[model_name] = {
                'status': 'loaded',
                'description': {
                    'openclip': 'OpenCLIP model (ViT-B/32) - better for general image search',
                    'clip': 'Original CLIP model (ViT-B/32) - good for general image search'
                }.get(model_name, 'No description available')
            }
        
        return jsonify({
            'status': 'success',
            'available_models': available_models,
            'default_model': 'openclip',
            'models': models_info
        })
        
    except Exception as e:
        import traceback
        app.logger.error(f"Error in list_models: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'available_models': list(faiss_handlers.keys())
        }), 500

@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    """Xóa cache và tải lại models"""
    try:
        global models, faiss_indexes, id_maps
        
        # Giải phóng bộ nhớ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Xóa models và indexes cũ
        models = {}
        faiss_indexes = {}
        id_maps = {}
        
        # Tải lại
        load_models_and_indexes()
        
        response = jsonify({"status": "Cache cleared and models reloaded"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        response = jsonify({"error": str(e)})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500
