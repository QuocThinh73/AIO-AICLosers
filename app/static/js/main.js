// ===================================================================
// MAIN.JS - Entry point và khởi tạo ứng dụng
// ===================================================================

// Import các modules (sẽ được tải qua script tags)
// - search.js: Xử lý tìm kiếm và hiển thị kết quả
// - models.js: Xử lý chọn models
// - objects.js: Xử lý chọn objects  
// - modal.js: Xử lý modal hiển thị ảnh chi tiết

// Biến toàn cục cho ứng dụng
window.AppData = {
    availableModels: [],
    availableObjects: [],
    selectedObjects: []
};

// Các phần tử DOM chính
window.AppElements = {
    searchForm: document.getElementById('searchForm'),
    searchInput: document.getElementById('searchInput'),
    ocrInput: document.getElementById('ocrInput'),
    topKInput: document.getElementById('topK'),
    resultsDiv: document.getElementById('results'),
    modelSelect: document.getElementById('modelSelect')
};

// Hàm khởi tạo ứng dụng
function initApp() {
    // Kiểm tra các phần tử cần thiết
    if (!window.AppElements.searchForm || !window.AppElements.searchInput || 
        !window.AppElements.resultsDiv || !window.AppElements.modelSelect) {
        return;
    }
    
    // Khởi tạo các modules
    if (window.SearchModule) {
        window.SearchModule.init();
    }
    
    if (window.ModelsModule) {
        window.ModelsModule.init();
    }
    
    if (window.ObjectsModule) {
        window.ObjectsModule.init();
    }
    
    if (window.ModalModule) {
        window.ModalModule.init();
    }
}

// Khởi động ứng dụng khi DOM đã tải xong
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}
