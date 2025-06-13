# Hướng dẫn chạy Flask API cho HCMAI2025

## Yêu cầu hệ thống
- Python 3.7+
- Đã cài đặt các thư viện trong `requirements.txt`

## Cài đặt

1. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Chạy ứng dụng

### 1. Khởi động FastAPI server (nếu chưa chạy)
```bash
uvicorn app.api:app --reload --port 8000
```

### 2. Khởi động Flask API
Mở một terminal khác và chạy:
```bash
python app_flask.py
```

## Sử dụng API

### 1. Kiểm tra trạng thái API
```
GET http://localhost:5000/health
```

### 2. Tìm kiếm ảnh bằng văn bản
```
POST http://localhost:5000/api/search
```

**Request body (JSON):**
```json
{
    "query": "mô tả ảnh cần tìm",
    "top_k": 5,
    "model_name": "clip"
}
```

**Response:**
```json
{
    "scores": [...],
    "indices": [...],
    "paths": [...]
}
```

## Các tham số

- `query` (bắt buộc): Mô tả văn bản để tìm kiếm ảnh
- `top_k` (tùy chọn, mặc định=5): Số lượng kết quả trả về
- `model_name` (tùy chọn, mặc định='clip'): Tên mô hình sử dụng ('clip' hoặc 'openclip')
