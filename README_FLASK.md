# HCMAI 2025 - Image Search with CLIP (Flask Version)

This project implements a text-to-image search system using CLIP and FAISS for the HCMAI 2025 challenge, with a Flask-based web interface.

## Cấu trúc dự án mới

```
.
├── app/                      # Ứng dụng Flask
│   ├── __init__.py           # Khởi tạo ứng dụng
│   ├── __main__.py           # Điểm vào chạy ứng dụng
│   └── app.py                # Logic chính của ứng dụng
├── data/                     # Thư mục dữ liệu
│   └── keyframes/            # Các khung hình chính từ video
├── database/                 # Chỉ mục FAISS và ánh xạ ID
├── static/                   # File tĩnh cho giao diện web
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── images/               # Ảnh mặc định
└── templates/                # Mẫu HTML
    └── index.html
```

## Yêu cầu hệ thống

- Python 3.7+
- CUDA (khuyến nghị nếu có GPU)
- Thư viện: Xem `requirements.txt`

## Cài đặt

1. Tạo môi trường ảo:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Trên Windows
   ```

2. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

3. Đảm bảo bạn đã có:
   - Thư mục `data/keyframes` chứa các ảnh cần tìm kiếm
   - Thư mục `database` chứa các file FAISS index:
     - `clip_faiss.bin`
     - `clip_id_map.json`
     - `openclip_faiss.bin`
     - `openclip_id_map.json`

## Chạy ứng dụng

1. Khởi động ứng dụng:
   ```bash
   python -m app
   ```
   hoặc
   ```bash
   python app/__main__.py
   ```

2. Mở trình duyệt và truy cập:
   ```
   http://localhost:5000
   ```

## Các API Endpoint

- `POST /api/search` - Tìm kiếm ảnh bằng văn bản
  ```json
  {
    "query": "một người đang đi xe đạp",
    "top_k": 12,
    "model_name": "clip"  # hoặc "openclip"
  }
  ```

- `GET /health` - Kiểm tra trạng thái ứng dụng
- `GET /api/models` - Liệt kê các model đã tải
- `POST /api/clear_cache` - Xóa cache và tải lại models

## Giao diện web

Giao diện web cho phép:
- Nhập từ khóa tìm kiếm
- Chọn giữa các mô hình (CLIP/OpenCLIP)
- Xem kết quả với điểm số tương đồng
- Xem ảnh kết quả trực tiếp trên giao diện

## Hướng dẫn phát triển

1. **Cấu trúc thư mục**:
   - `app/` chứa toàn bộ mã nguồn Python
   - `static/` chứa CSS, JavaScript và ảnh tĩnh
   - `templates/` chứa các file HTML

2. **Thêm model mới**:
   - Thêm code tải model trong hàm `load_models_and_indexes()`
   - Thêm hàm xử lý embedding tương ứng

3. **Gỡ lỗi**:
   - Bật chế độ debug: `app.run(debug=True)`
   - Xem log trong console để biết chi tiết lỗi

## Giấy phép

Dự án này được cấp phép theo giấy phép MIT - xem file [LICENSE](LICENSE) để biết thêm chi tiết.
