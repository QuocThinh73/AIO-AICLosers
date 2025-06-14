// Biến toàn cục
let availableModels = [];
const searchForm = document.getElementById('searchForm');
const searchInput = document.getElementById('searchInput');
const resultsDiv = document.getElementById('results');
const loadingDiv = document.querySelector('.loading');
const modelSelect = document.getElementById('modelSelect');
const modelInfo = document.getElementById('modelInfo');

// Hàm chính
function initApp() {
    console.log('Khởi tạo ứng dụng...');
    
    // Ẩn loading khi khởi tạo
    if (loadingDiv) loadingDiv.style.display = 'none';
    
    // Kiểm tra các phần tử cần thiết
    if (!searchForm || !searchInput || !resultsDiv || !modelSelect || !modelInfo) {
        console.error('Không tìm thấy các phần tử cần thiết trên trang');
        return;
    }
    
    // Thêm sự kiện submit form
    searchForm.addEventListener('submit', handleSearch);
    
    // Tải danh sách model
    loadModels();
}

// Hàm tải danh sách model
function loadModels() {
    console.log('Đang tải danh sách model...');
    
    fetch('/api/models')
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error || `HTTP error! status: ${response.status}`);
                }).catch(() => {
                    throw new Error(`HTTP error! status: ${response.status}`);
                });
            }
            return response.json();
        })
        .then(data => {
            console.log('Dữ liệu model:', data);
            
            // Lấy danh sách model từ phản hồi
            availableModels = data.available_models || [];
            
            if (availableModels.length === 0) {
                console.warn('Không có model nào khả dụng');
                modelInfo.textContent = 'Không tìm thấy model nào';
                modelInfo.style.color = 'orange';
            } else {
                updateModelSelect();
            }
        })
        .catch(error => {
            console.error('Lỗi khi tải model:', error);
            modelInfo.textContent = 'Lỗi khi tải danh sách model';
            modelInfo.style.color = 'red';
        });
}

// Cập nhật dropdown chọn model
function updateModelSelect() {
    console.log('Cập nhật dropdown với models:', availableModels);
    
    modelSelect.innerHTML = ''; // Xóa các option cũ
    
    // Thêm option mặc định
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = '-- Chọn model --';
    defaultOption.disabled = true;
    defaultOption.selected = true;
    modelSelect.appendChild(defaultOption);
    
    // Thêm các model vào dropdown
    availableModels.forEach(model => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = model;
        modelSelect.appendChild(option);
    });
    
    // Cập nhật thông tin model
    updateModelInfo(availableModels[0]);
}

// Cập nhật thông tin model
function updateModelInfo(modelName) {
    if (!modelName) return;
    
    const modelInfoMap = {
        'blip2': 'BLIP-2: Mô hình đa phương tiện mạnh mẽ từ Salesforce',
        'openclip': 'OpenCLIP: Phiên bản mở của CLIP',
        'clip': 'CLIP: Mô hình đa phương tiện từ OpenAI'
    };
    
    modelInfo.textContent = modelInfoMap[modelName] || `Model: ${modelName}`;
    modelInfo.style.color = '#666';
}

// Lấy giá trị top-k từ thanh trượt
const topKSlider = document.getElementById('topK');
const topKValue = document.getElementById('topKValue');

// Cập nhật giá trị hiển thị khi thanh trượt thay đổi
if (topKSlider && topKValue) {
    topKSlider.addEventListener('input', function() {
        topKValue.textContent = this.value;
    });
}

// Xử lý tìm kiếm
function handleSearch(e) {
    e.preventDefault();
    
    const query = searchInput.value.trim();
    const model = modelSelect.value;
    const topK = topKSlider ? parseInt(topKSlider.value) : 10;
    
    // Validate input
    if (!query) {
        alert('Vui lòng nhập từ khóa tìm kiếm');
        return;
    }
    
    if (!model) {
        alert('Vui lòng chọn model');
        return;
    }
    
    console.log('Đang tìm kiếm:', { query, model, topK });
    
    // Hiển thị loading
    if (loadingDiv) loadingDiv.style.display = 'flex';
    resultsDiv.innerHTML = '';
    
    // Tạo controller để có thể hủy request nếu cần
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000); // Timeout sau 30s
    
    // Kiểm tra query hợp lệ
    if (!query || typeof query !== 'string' || query.trim() === '') {
        throw new Error('Vui lòng nhập từ khóa tìm kiếm hợp lệ');
    }
    
    // Tạo URL với tham số để tránh CORS preflight
    const url = new URL('http://localhost:5000/api/search');
    
    // Thêm tham số vào URL
    url.searchParams.append('query', query);
    url.searchParams.append('model_name', model);
    url.searchParams.append('top_k', topK.toString());
    
    // Gọi API tìm kiếm với GET
    fetch(url.toString(), {
        method: 'GET',
        headers: {
            'Accept': 'application/json'
            // Không thêm header không cần thiết
        },
        signal: controller.signal
    })
    .then(async response => {
        clearTimeout(timeoutId); // Clear timeout nếu request thành công
        
        // Kiểm tra nếu response không phải là JSON
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            const text = await response.text();
            console.error('Phản hồi không phải JSON:', text);
            throw new Error(`Server trả về lỗi không xác định (${response.status}): ${text.substring(0, 200)}`);
        }
        
        const data = await response.json().catch(err => {
            console.error('Lỗi khi parse JSON:', err);
            throw new Error('Không thể đọc phản hồi từ máy chủ');
        });
        
        if (!response.ok) {
            console.error('Lỗi từ server:', data);
            const errorMessage = data.error || data.detail || data.message || `Lỗi HTTP: ${response.status}`;
            throw new Error(errorMessage, { cause: JSON.stringify(data, null, 2) });
        }
        
        return data;
    })
    .then(displayResults)
    .catch(error => {
        console.error('Lỗi khi tìm kiếm:', error);
        
        // Tạo thông báo lỗi chi tiết
        let errorMessage = error.message || 'Có lỗi xảy ra khi tìm kiếm';
        
        // Kiểm tra loại lỗi
        if (error.name === 'AbortError') {
            errorMessage = 'Yêu cầu tìm kiếm đã hết thời gian chờ (30s). Vui lòng thử lại.';
        } else if (error.message.includes('Failed to fetch')) {
            errorMessage = 'Không thể kết nối đến máy chủ. Vui lòng kiểm tra kết nối mạng và thử lại.';
        }
        
        // Thêm thông tin bổ sung nếu có
        if (error.cause) {
            errorMessage += `\n\nChi tiết lỗi:\n${error.cause}`;
        }
        
        resultsDiv.innerHTML = `
            <div class="error" style="color: #721c24; padding: 15px; margin: 15px 0; border: 1px solid #f5c6cb; background-color: #f8d7da; border-radius: 4px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <i class="fas fa-exclamation-triangle" style="font-size: 1.2em;"></i>
                    <h3 style="margin: 0; font-size: 1.1em;">Lỗi khi tìm kiếm</h3>
                </div>
                <div style="background: white; padding: 10px; border-radius: 4px; font-family: monospace; white-space: pre-wrap; word-break: break-word;">
                    ${errorMessage}
                </div>
                <p style="margin-top: 10px; margin-bottom: 0; font-size: 0.9em;">
                    Vui lòng thử lại hoặc liên hệ quản trị viên nếu lỗi vẫn còn.
                </p>
            </div>
        `;
    })
    .finally(() => {
        if (loadingDiv) loadingDiv.style.display = 'none';
    });
}

// Hiển thị kết quả tìm kiếm
function displayResults(data) {
    console.log('Kết quả tìm kiếm:', data);
    
    if (!data || !data.paths || data.paths.length === 0) {
        resultsDiv.innerHTML = `
            <div class="no-results" style="text-align: center; padding: 20px; color: #666;">
                <i class="fas fa-search" style="font-size: 2em; margin-bottom: 10px; opacity: 0.5;"></i>
                <p>Không tìm thấy kết quả phù hợp với "${data?.query || 'từ khóa'}"</p>
            </div>
        `;
        return;
    }
    
    let html = `
        <div class="search-info" style="margin-bottom: 20px; padding: 10px; background: #f8f9fa; border-radius: 4px;">
            <p>Kết quả tìm kiếm cho: <strong>${data.query || 'Không có từ khóa'}</strong></p>
            <p>Sử dụng model: <strong>${data.model || 'Không xác định'}</strong></p>
            <p>Tìm thấy <strong>${data.paths.length}</strong> kết quả</p>
        </div>
        <div class="image-grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px;">
    `;
    
    data.paths.forEach((path, index) => {
        const score = data.scores && data.scores[index] !== undefined 
            ? (data.scores[index] * 100).toFixed(1) + '%' 
            : '';
            
        // Lấy tên file từ filename nếu có, nếu không thì từ path
        const filename = data.filenames && data.filenames[index] 
            ? data.filenames[index] 
            : path.split('/').pop();
            
        // Create image path - handle different path formats
        let imagePath;
        try {
            // Get just the filename without any path
            const filename = path.split('/').pop();
            
            // Always use the filename in the path to avoid any path traversal issues
            imagePath = `/data/keyframes/${filename}`;
            
            console.log(`Original path: ${path}, Using image path: ${imagePath}`);
        } catch (e) {
            console.error('Error constructing image path:', e);
            imagePath = '/static/images/no-image.jpg';
        }
        
        html += `
            <div class="image-item" style="border: 1px solid #ddd; border-radius: 4px; overflow: hidden; display: flex; flex-direction: column; height: 100%;">
                <div class="image-container" style="position: relative; padding-top: 100%;">
                    <img 
                        src="${imagePath}" 
                        alt="Kết quả ${index + 1}"
                        onerror="this.onerror=null; this.src='/static/images/no-image.jpg'"
                        loading="lazy"
                        style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover;"
                    >
                    ${score ? `<div class="score" style="position: absolute; bottom: 5px; right: 5px; background: rgba(0,0,0,0.7); color: white; padding: 2px 5px; border-radius: 3px; font-size: 12px;">${score}</div>` : ''}
                </div>
                <div class="image-info" style="padding: 8px; font-size: 12px; background: #f8f9fa; border-top: 1px solid #eee; flex-grow: 1; display: flex; flex-direction: column; justify-content: space-between;">
                    <div class="image-filename" style="font-weight: bold; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${filename}">
                        ${filename}
                    </div>
                    <div class="image-path" style="color: #666; font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="${path}">
                        ${path}
                    </div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    resultsDiv.innerHTML = html;
}

// Chạy ứng dụng khi DOM đã tải xong
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}
