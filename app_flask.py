import os
import json
import requests
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# Khởi tạo Flask app
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
CORS(app)  # Cho phép CORS cho tất cả các route

# Cấu hình
app.config['UPLOAD_FOLDER'] = 'static/images'
app.config['DATA_FOLDER'] = 'data'
API_URL = "http://127.0.0.1:8000"  # Địa chỉ của FastAPI server

# Tạo thư mục nếu chưa tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['DATA_FOLDER'], 'keyframes'), exist_ok=True)

# Route chính - hiển thị giao diện web
@app.route('/')
def index():
    return render_template('index.html')

# Phục vụ các file tĩnh
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# Route để phục vụ ảnh từ thư mục data/keyframes
@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory(app.config['DATA_FOLDER'], filename)

@app.route('/api/search', methods=['POST'])
def search():
    try:
        # Lấy dữ liệu từ request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        query = data.get('query', '')
        top_k = data.get('top_k', 12)
        model_name = data.get('model_name', 'clip')
        
        print(f"\n[DEBUG] Received search request:")
        print(f"- Query: {query}")
        print(f"- Top K: {top_k}")
        print(f"- Model: {model_name}")
        
        # Kiểm tra dữ liệu đầu vào
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        if model_name not in ["clip", "openclip"]:
            return jsonify({"error": "Invalid model_name. Choose 'clip' or 'openclip'"}), 400
        
        # Gửi yêu cầu đến FastAPI server
        try:
            response = requests.post(
                f"{API_URL}/text-search/",
                json={
                    "query": query,
                    "top_k": int(top_k),
                    "model_name": model_name
                },
                timeout=30  # Timeout sau 30 giây
            )
            response.raise_for_status()  # Ném ra lỗi nếu status code không phải 2xx
            
            # Log kết quả (chỉ log 1-2 kết quả đầu tiên để tránh log quá dài)
            result = response.json()
            print(f"[DEBUG] Search successful. Found {len(result.get('paths', []))} results")
            if 'paths' in result and len(result['paths']) > 0:
                print(f"First result path: {result['paths'][0]}")
            
            return jsonify(result)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to connect to model server at {API_URL}: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return jsonify({
                "error": "Could not connect to the model server",
                "details": str(e),
                "api_url": API_URL
            }), 503
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] Unexpected error: {error_trace}")
        return jsonify({
            "error": "An unexpected error occurred",
            "details": str(e),
            "trace": error_trace if app.debug else None
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint để kiểm tra trạng thái của API"""
    return jsonify({"status": "healthy", "service": "flask-api"})

if __name__ == '__main__':
    # Chạy server Flask
    app.run(host='0.0.0.0', port=5000, debug=True)
