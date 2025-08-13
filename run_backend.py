from backend.app.main import create_app

# Thiết lập biến môi trường để khắc phục lỗi OpenMP trước khi import các thư viện khác
# Lưu ý: Đây là giải pháp tạm thời, không nên dùng trong môi trường production
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

app = create_app()

if __name__ == "__main__":
    import uvicorn
    
    # Khởi động Uvicorn server
    uvicorn.run(
        "run_backend:app",
        host="0.0.0.0",  # Cho phép kết nối từ bên ngoài
        port=8000,
        reload=False     # Tắt auto-reload để tránh lỗi khi phát triển
    )
