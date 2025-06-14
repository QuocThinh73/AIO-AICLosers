import os
import json
import faiss
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image
import torch
import clip
import open_clip

# Khởi tạo Flask app
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Cấu hình CORS chi tiết hơn
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
    UPLOAD_FOLDER='static/images',
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
faiss_indexes = {}
id_maps = {}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_models_and_indexes():
    """Tải các model và FAISS indexes từ thư mục database"""
    global models, faiss_indexes, id_maps
    
    # In thông tin thư mục database (sử dụng encoding phù hợp cho Windows)
    import sys
    import io
    # Đặt lại stdout để hỗ trợ in Unicode trên Windows
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    
    database_path = os.path.abspath(app.config['DATABASE_FOLDER'])
    print("\n" + "="*50)
    print(f"Đang tải chỉ mục từ thư mục: {database_path}")
    
    # Kiểm tra xem thư mục tồn tại không
    if not os.path.exists(database_path):
        print(f"LỖI: Thư mục {database_path} không tồn tại")
        return
        
    # Liệt kê tất cả file trong thư mục
    try:
        # Khởi tạo các models
        models = {}

        # Tải OpenCLIP model
        print("\nĐang tải OpenCLIP model...")
        try:
            openclip_model, _, openclip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
            openclip_model = openclip_model.to(device)
            openclip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
            models['openclip'] = (openclip_model, openclip_preprocess, openclip_tokenizer)
            print("Đã tải xong OpenCLIP model")
        except Exception as e:
            print(f"Không thể tải OpenCLIP model: {str(e)}")
            
        # Tải CLIP model
        print("\nĐang tải CLIP model...")
        try:
            clip_model, clip_preprocess = clip.load('ViT-B/32', device=device)
            models['clip'] = (clip_model, clip_preprocess)
            print("Đã tải xong CLIP model")
        except Exception as e:
            print(f"Không thể tải CLIP model: {str(e)}")

        # In danh sách các model đã tải thành công
        if models:
            print("\nCác model đã tải thành công:")
            for name in models.keys():
                print(f"- {name}")
        else:
            print("\nCảnh báo: Không có model nào được tải thành công")

        # Liệt kê tất cả file trong thư mục
        database_files = os.listdir(database_path)
        print(f"Tìm thấy {len(database_files)} file trong thư mục database:")
        for f in database_files:
            print(f"- {f}")
    except Exception as e:
        print(f"Lỗi khi đọc thư mục database: {str(e)}")
        return
    
    # Tìm tất cả các file index trong thư mục database
    available_models = set()
    for filename in database_files:
        if filename.endswith('_faiss.bin'):
            model_name = filename.replace('_faiss.bin', '')
            available_models.add(model_name)
    
    print(f"\nTìm thấy các model có sẵn: {available_models}")
    
    if not available_models:
        print("Cảnh báo: Không tìm thấy file FAISS index nào trong thư mục database")
        print("Vui lòng đảm bảo có ít nhất một file có đuôi '_faiss.bin'")
    
    # Chỉ tải các model có đầy đủ file cần thiết
    for model_name in available_models:
        index_path = os.path.join(database_path, f'{model_name}_faiss.bin')
        print(f"\nĐang xử lý model: {model_name}")
        print(f"- Đường dẫn index: {index_path}")
        
        # Thử tìm file id_map với các định dạng khác nhau
        possible_id_maps = [
            f'{model_name}_id_map.json',
            f'{model_name}_id2path.json',
            f'{model_name}_id2path.pkl',
            f'{model_name}_map.json',
            f'{model_name}_mapping.json'
        ]
        
        id_map_path = None
        for possible_map in possible_id_maps:
            full_path = os.path.join(database_path, possible_map)
            if os.path.exists(full_path):
                id_map_path = full_path
                print(f"- Tìm thấy file ánh xạ: {possible_map}")
                break
        
        if not id_map_path:
            print(f"Cảnh báo: Không tìm thấy file ánh xạ cho model {model_name}")
            print("Đã thử các định dạng:", ", ".join(possible_id_maps))
            continue
            
        try:
            # Tải FAISS index
            print(f"- Đang tải FAISS index từ: {os.path.basename(index_path)}")
            faiss_indexes[model_name] = faiss.read_index(index_path)
            print(f"  -> Đã tải xong FAISS index")
            
            # Tải ID map với hỗ trợ nhiều định dạng
            print(f"- Đang tải file ánh xạ từ: {os.path.basename(id_map_path)}")
            if id_map_path.endswith('.pkl'):
                import pickle
                with open(id_map_path, 'rb') as f:
                    id_maps[model_name] = pickle.load(f)
                # Nếu là dict với key là số, chuyển sang string để đồng bộ
                if id_maps[model_name] and isinstance(next(iter(id_maps[model_name].keys())), (int, np.integer)):
                    id_maps[model_name] = {str(k): v for k, v in id_maps[model_name].items()}
            else:
                with open(id_map_path, 'r', encoding='utf-8') as f:
                    id_maps[model_name] = json.load(f)
            
            print(f"  -> Đã tải xong file ánh xạ, tổng cộng {len(id_maps[model_name])} ảnh")
            
        except Exception as e:
            import traceback
            print(f"Lỗi khi tải {model_name}:\n{traceback.format_exc()}")
            if model_name in faiss_indexes:
                del faiss_indexes[model_name]
            if model_name in id_maps:
                del id_maps[model_name]
    
    # In thông tin các model đã tải
    print("\n" + "="*50)
    print("TÓM TẮT CÁC MODEL ĐÃ TẢI:")
    if not faiss_indexes:
        print("KHÔNG CÓ MODEL NÀO ĐƯỢC TẢI THÀNH CÔNG")
    else:
        for model_name in faiss_indexes.keys():
            print(f"- {model_name}: {len(id_maps.get(model_name, {}))} ảnh")
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

def get_text_embedding(text, model_name=None):
    """Lấy embedding cho văn bản sử dụng model đã chọn"""
    if not text:
        raise ValueError("Vui lòng nhập văn bản để tìm kiếm")
        
    if not models:
        raise RuntimeError("Không có model nào được tải. Vui lòng kiểm tra lại quá trình khởi tạo.")
    
    # Nếu không chỉ định model_name, sử dụng model đầu tiên
    if model_name is None:
        model_name = next(iter(models.keys()))
    
    if model_name not in models:
        available_models = list(models.keys())
        raise ValueError(f"Model '{model_name}' không tồn tại. Các model khả dụng: {available_models}")
    
    try:
        print(f"Đang xử lý văn bản với model: {model_name}")
        
        if model_name == 'openclip':
            # Xử lý OpenCLIP model
            model, _, tokenizer = models[model_name]
            text_tokens = tokenizer([text]).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                
        elif model_name == 'clip':
            # Xử lý CLIP model
            model, _ = models[model_name]
            text_tokens = clip.tokenize([text], truncate=True).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
        else:
            raise ValueError(f"Model {model_name} không được hỗ trợ")
        
        # Chuyển đổi sang numpy array và đảm bảo kiểu float32
        if isinstance(text_features, torch.Tensor):
            text_features = text_features.cpu().numpy()
        
        # Đảm bảo kết quả là 2D array (batch_size, embedding_dim)
        if len(text_features.shape) == 1:
            text_features = text_features.reshape(1, -1)
            
        print(f"Tạo embedding thành công với shape: {text_features.shape}")
        return text_features.astype('float32')
        
    except Exception as e:
        import traceback
        error_msg = f"Lỗi trong get_text_embedding (model={model_name}): {str(e)}\n{traceback.format_exc()}"
        print(f"[LỖI] {error_msg}")
        raise

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
        # Sử dụng OpenCLIP làm mặc định
        model_name = data.get('model_name', 'openclip')
        top_k = min(int(data.get('top_k', 12)), 50)  # Giới hạn tối đa 50 kết quả

        if not query:
            app.logger.error('No query provided')
            return jsonify({'error': 'Query is required'}), 400

        app.logger.info(f'Search params - query: {query}, model: {model_name}, top_k: {top_k}')

        # Kiểm tra model có tồn tại không
        available_models = list(faiss_indexes.keys())
        if model_name not in available_models:
            error_msg = f'Model {model_name} not found. Available models: {available_models}'
            app.logger.error(error_msg)
            return jsonify({
                'error': error_msg,
                'available_models': available_models
            }), 400

        # Kiểm tra xem model có được load không
        loaded_models = list(models.keys())
        if model_name not in loaded_models:
            error_msg = f'Model {model_name} is not loaded. Available models: {loaded_models}'
            app.logger.error(error_msg)
            return jsonify({
                'error': error_msg,
                'available_models': loaded_models
            }), 500

        # Kiểm tra xem có ID map cho model không
        if model_name not in id_maps or not id_maps[model_name]:
            error_msg = f'No ID map found for model {model_name}'
            app.logger.error(error_msg)
            return jsonify({'error': error_msg}), 500

        # Lấy embedding cho query
        app.logger.info(f'Getting text embedding for query: {query}')
        try:
            query_embedding = get_text_embedding(query, model_name)
            app.logger.info(f'Successfully got text embedding. Shape: {query_embedding.shape if hasattr(query_embedding, "shape") else "N/A"}')
        except Exception as e:
            error_msg = f'Error getting text embedding: {str(e)}'
            app.logger.error(error_msg, exc_info=True)
            return jsonify({'error': error_msg}), 500
        
        # Kiểm tra FAISS index
        if faiss_indexes.get(model_name) is None:
            error_msg = f'FAISS index for model {model_name} is not loaded'
            app.logger.error(error_msg)
            return jsonify({'error': error_msg}), 500
            
        # Tìm kiếm trong FAISS
        app.logger.info('Searching in FAISS index...')
        try:
            # Đảm bảo query_embedding có đúng shape (1, embedding_dim)
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Chuyển đổi sang mảng numpy nếu cần
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding, dtype=np.float32)
            
            # Thực hiện tìm kiếm
            distances, indices = faiss_indexes[model_name].search(query_embedding, top_k)
            app.logger.info(f'FAISS search completed. Found {len(indices[0])} results')
            
            # Đổi tên biến để nhất quán với phần code bên dưới
            D, I = distances, indices
            
        except Exception as e:
            error_msg = f'Error during FAISS search: {str(e)}'
            app.logger.error(error_msg, exc_info=True)
            return jsonify({'error': error_msg, 'details': str(e)}), 500
        
        # Lấy đường dẫn ảnh từ id_map
        id_map = id_maps.get(model_name, {})
        if not id_map:
            app.logger.warning(f'No ID map found for model {model_name}')
        
        results = []
        valid_results = 0
        
        for idx, score in zip(I[0], D[0]):
            if str(idx) in id_map and idx != -1:  # -1 có thể là kết quả không hợp lệ
                results.append({
                    'path': id_map[str(idx)],
                    'score': float(score)  # Chuyển numpy.float32 thành float thông thường
                })
                valid_results += 1
        
        app.logger.info(f'Found {valid_results} valid results out of {len(I[0])} total results')
        
        # Sắp xếp theo score (từ thấp đến cao vì đây là khoảng cách)
        results.sort(key=lambda x: x['score'])
        
        # Kiểm tra xem có kết quả nào không
        if not results:
            app.logger.warning('No valid results found')
            response_data = {
                'query': query,
                'model': model_name,
                'message': 'No results found',
                'paths': [],
                'scores': []
            }
            response = jsonify(response_data)
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            return response
        
        # Trả về kết quả với CORS headers
        response_data = {
            'query': query,
            'model': model_name,
            'paths': [r['path'].replace('data/', '', 1) for r in results],  # Remove the first occurrence of 'data/'
            'scores': [r['score'] for r in results],
            'filenames': [os.path.basename(r['path']) for r in results]  # Add filenames for display
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
        # Lấy danh sách các model có sẵn trong thư mục database
        available_models = list(faiss_indexes.keys())
        
        # Lấy danh sách các model đã được tải vào bộ nhớ
        loaded_models = list(models.keys())
        
        # Tạo thông tin chi tiết về từng model
        models_info = {}
        for model_name in available_models:
            models_info[model_name] = {
                'status': 'loaded' if model_name in loaded_models else 'not_loaded',
                'index': 'available' if model_name in faiss_indexes and faiss_indexes[model_name] is not None else 'missing',
                'id_map': 'available' if model_name in id_maps and id_maps[model_name] else 'missing',
                'num_images': len(id_maps.get(model_name, {})),
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
            'available_models': list(faiss_indexes.keys())
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
