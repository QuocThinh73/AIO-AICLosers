import sys
import os
import logging
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from backend.app.api.endpoints import search
from backend.app.services.database_service import database_service
from backend.app.core.config import settings

# Flags để theo dõi tình trạng của các service
ELASTICSEARCH_AVAILABLE = False
QDRANT_AVAILABLE = False
FAISS_AVAILABLE = False

def validate_startup():
    """Kiểm tra tình trạng của Elasticsearch, Qdrant và FAISS trước khi khởi chạy."""
    global ELASTICSEARCH_AVAILABLE, QDRANT_AVAILABLE, FAISS_AVAILABLE
    
    # Kiểm tra FAISS embeddings
    embedding_models = database_service.get_available_embedding_models()
    if not embedding_models:
        logging.error("FATAL: Không tìm thấy FAISS embedding files! Ứng dụng sẽ dừng.")
        sys.exit(1)
    else:
        FAISS_AVAILABLE = True
        logging.info(f"Tìm thấy {len(embedding_models)} embedding models: {', '.join(embedding_models)}")
        
        # Nếu một số mô hình không tìm thấy, chỉ cảnh báo chứ không thoát
        # So sánh với danh sách cấu hình từ settings
        configured_models = list(settings.EMBEDDING_MODELS.keys())
        if len(embedding_models) < len(configured_models):
            missing_models = set(configured_models) - set(embedding_models)
            logging.warning(f"Không tìm thấy {len(missing_models)} mô hình embedding cấu hình: {', '.join(missing_models)}")
            logging.warning("Các tính năng liên quan đến các mô hình này sẽ bị vô hiệu hóa.")
    
    # Kiểm tra Elasticsearch
    es_client = database_service.get_elasticsearch_client()
    if es_client is None:
        logging.warning("Elasticsearch không khả dụng. Tính năng object detection search và OCR search sẽ bị vô hiệu hóa.")
        ELASTICSEARCH_AVAILABLE = False
    else:
        ELASTICSEARCH_AVAILABLE = True
        logging.info("Elasticsearch đang chạy và sẵn sàng.")
    
    # Kiểm tra Qdrant
    qdrant_client = database_service.get_qdrant_client()
    if qdrant_client is None:
        logging.warning("Qdrant không khả dụng. Tính năng captioning search sẽ bị vô hiệu hóa.")
        QDRANT_AVAILABLE = False
    else:
        QDRANT_AVAILABLE = True
        logging.info("Qdrant đang chạy và sẵn sàng.")
    
    # Nếu cả Elasticsearch và Qdrant đều không khả dụng
    if not ELASTICSEARCH_AVAILABLE and not QDRANT_AVAILABLE:
        logging.error("FATAL: Cả Elasticsearch và Qdrant đều không khả dụng! Ứng dụng sẽ dừng.")
        sys.exit(1)
    
    # Hiển thị tình trạng tổng quan
    logging.info(f"Tình trạng dịch vụ: FAISS={FAISS_AVAILABLE}, Elasticsearch={ELASTICSEARCH_AVAILABLE}, Qdrant={QDRANT_AVAILABLE}")
    return True

def create_app() -> FastAPI:
    # Kiểm tra tình trạng trước khi khởi tạo app
    validate_startup()
    
    app = FastAPI()
    app.include_router(search.router, prefix="/api/search", tags=["search"])
    
    # Middleware để kiểm tra các tính năng không khả dụng
    @app.middleware("http")
    async def check_service_availability(request: Request, call_next):
        # Kiểm tra các request liên quan đến search
        if request.url.path.startswith("/api/search/base_search") and request.method == "POST":
            try:
                # Parse request body để kiểm tra các tính năng được yêu cầu
                body = await request.json()
                
                # Kiểm tra các tính năng không khả dụng
                if not ELASTICSEARCH_AVAILABLE and (body.get("object_detection", False) or body.get("ocr", False)):
                    return JSONResponse(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        content={"detail": "Elasticsearch không khả dụng. Object detection và OCR search không thể sử dụng."}
                    )
                    
                if not QDRANT_AVAILABLE and body.get("captioning", False):
                    return JSONResponse(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        content={"detail": "Qdrant không khả dụng. Captioning search không thể sử dụng."}
                    )
            except Exception:
                # Nếu không parse được body, tiếp tục xử lý request bình thường
                pass
        
        response = await call_next(request)
        return response
    
    return app

app = create_app()