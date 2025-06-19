// Biến toàn cục
let availableModels = [];
let availableObjects = [];
let selectedObjects = [];
let filteredObjects = [];
const searchForm = document.getElementById('searchForm');
const searchInput = document.getElementById('searchInput');
const ocrInput = document.getElementById('ocrInput');
const resultsDiv = document.getElementById('results');
const loadingDiv = document.querySelector('.loading');
const modelSelect = document.getElementById('modelSelect');
const loadingObjects = document.querySelector('.loading-objects');

// Object selector elements
const objectSelectorBtn = document.getElementById('objectSelectorBtn');
const objectDropdown = document.getElementById('objectDropdown');
const objectSearch = document.getElementById('objectSearch');
const objectList = document.getElementById('objectList');
const selectedObjectsDisplay = document.getElementById('selectedObjectsDisplay');
const selectedObjectsList = document.getElementById('selectedObjectsList');
const clearSelectionBtn = document.getElementById('clearSelectionBtn');
const selectAllObjectsBtn = document.getElementById('selectAllObjectsBtn');
const clearAllObjectsBtn = document.getElementById('clearAllObjectsBtn');
const confirmSelectionBtn = document.getElementById('confirmSelectionBtn');

// Hàm chính
function initApp() {
    // Ẩn loading khi khởi tạo
    if (loadingDiv) loadingDiv.style.display = 'none';
    if (loadingObjects) loadingObjects.style.display = 'none';
    
    // Kiểm tra các phần tử cần thiết
    if (!searchForm || !searchInput || !resultsDiv || !modelSelect) {
        console.error('Không tìm thấy các phần tử cần thiết trên trang');
        return;
    }
    
    // Thêm sự kiện submit form
    searchForm.addEventListener('submit', handleSearch);
    
    // Thêm event listeners cho object selector
    setupObjectSelectorEvents();
    
    // Tải danh sách model và objects
    loadModels();
    loadObjects();
}

// Hàm tải danh sách model
function loadModels() {
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
            // Lấy danh sách model từ phản hồi
            availableModels = data.models || [];
            
            if (availableModels.length === 0) {
                console.warn('Không có model nào khả dụng');
            } else {
                updateModelSelect();
            }
        })
        .catch(error => {
            console.error('Lỗi khi tải model:', error);
            modelSelect.innerHTML = `
                <div class="error-message" style="color: #721c24; padding: 10px; background: #f8d7da; border-radius: 4px; font-size: 0.9rem;">
                    <i class="fas fa-exclamation-triangle"></i> Không thể tải danh sách models: ${error.message}
                </div>
            `;
        });
}

// Hàm tải danh sách objects
function loadObjects() {
    if (loadingObjects) loadingObjects.style.display = 'flex';
    
    fetch('/api/objects')
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
            // Lấy danh sách objects từ phản hồi
            availableObjects = data.objects || data || [];
            filteredObjects = [...availableObjects];
            
            if (availableObjects.length === 0) {
                console.warn('Không có object nào khả dụng');
                if (objectList) {
                    objectList.innerHTML = `
                        <div class="no-objects-message">
                            Không có objects nào khả dụng
                        </div>
                    `;
                }
            } else {
                updateObjectList();
            }
        })
        .catch(error => {
            console.error('Lỗi khi tải objects:', error);
            if (objectList) {
                objectList.innerHTML = `
                    <div class="error-message" style="color: #721c24; padding: 10px; background: #f8d7da; border-radius: 4px; font-size: 0.9rem;">
                        <i class="fas fa-exclamation-triangle"></i> Không thể tải danh sách objects: ${error.message}
                    </div>
                `;
            }
        })
        .finally(() => {
            if (loadingObjects) loadingObjects.style.display = 'none';
        });
}

function updateModelSelect() {
    const modelSelect = document.getElementById('modelSelect');
    modelSelect.innerHTML = ''; 
    
    // Thêm các model vào checkbox
    availableModels.forEach(model => {
        const label = document.createElement('label');
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.value = model;
        checkbox.name = 'models';
        checkbox.checked = true; // Tự động tick chọn mặc định

        const span = document.createElement('span');
        span.textContent = model;

        label.appendChild(checkbox);
        label.appendChild(span);
        modelSelect.appendChild(label);
    });
}

// Setup event listeners cho object selector
function setupObjectSelectorEvents() {
    // Toggle dropdown
    if (objectSelectorBtn) {
        objectSelectorBtn.addEventListener('click', toggleObjectDropdown);
    }
    
    // Search functionality
    if (objectSearch) {
        objectSearch.addEventListener('input', handleObjectSearch);
    }
    
    // Clear selection
    if (clearSelectionBtn) {
        clearSelectionBtn.addEventListener('click', clearObjectSelection);
    }
    
    // Bulk actions
    if (selectAllObjectsBtn) {
        selectAllObjectsBtn.addEventListener('click', () => selectAllObjects(true));
    }
    
    if (clearAllObjectsBtn) {
        clearAllObjectsBtn.addEventListener('click', () => selectAllObjects(false));
    }
    
    // Confirm selection
    if (confirmSelectionBtn) {
        confirmSelectionBtn.addEventListener('click', confirmObjectSelection);
    }
    
    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (objectDropdown && !objectDropdown.contains(e.target) && !objectSelectorBtn.contains(e.target)) {
            closeObjectDropdown();
        }
    });
}

// Toggle dropdown visibility
function toggleObjectDropdown() {
    if (objectDropdown) {
        const isVisible = objectDropdown.style.display === 'flex';
        if (isVisible) {
            closeObjectDropdown();
        } else {
            openObjectDropdown();
        }
    }
}

function openObjectDropdown() {
    if (objectDropdown && objectSelectorBtn) {
        objectDropdown.style.display = 'flex';
        objectSelectorBtn.classList.add('active');
        if (objectSearch) {
            objectSearch.focus();
        }
    }
}

function closeObjectDropdown() {
    if (objectDropdown && objectSelectorBtn) {
        objectDropdown.style.display = 'none';
        objectSelectorBtn.classList.remove('active');
        if (objectSearch) {
            objectSearch.value = '';
            filteredObjects = [...availableObjects];
            updateObjectList();
        }
    }
}

// Handle object search
function handleObjectSearch(e) {
    const searchTerm = e.target.value.toLowerCase().trim();
    
    if (searchTerm === '') {
        filteredObjects = [...availableObjects];
    } else {
        filteredObjects = availableObjects.filter(object => 
            object.toLowerCase().includes(searchTerm)
        );
    }
    
    updateObjectList();
}

// Update object list in dropdown
function updateObjectList() {
    if (!objectList) return;
    
    objectList.innerHTML = '';
    
    if (filteredObjects.length === 0) {
        objectList.innerHTML = `
            <div class="no-objects-message">
                Không tìm thấy object nào
            </div>
        `;
        return;
    }
    
    filteredObjects.forEach(object => {
        const item = document.createElement('div');
        item.className = 'object-list-item';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.value = object;
        checkbox.checked = selectedObjects.includes(object);
        checkbox.addEventListener('change', (e) => handleObjectSelection(e, object));
        
        const label = document.createElement('label');
        label.textContent = object;
        label.addEventListener('click', () => {
            checkbox.checked = !checkbox.checked;
            handleObjectSelection({ target: checkbox }, object);
        });
        
        item.appendChild(checkbox);
        item.appendChild(label);
        
        if (selectedObjects.includes(object)) {
            item.classList.add('selected');
        }
        
        objectList.appendChild(item);
    });
}

// Handle individual object selection
function handleObjectSelection(e, object) {
    if (e.target.checked) {
        if (!selectedObjects.includes(object)) {
            selectedObjects.push(object);
        }
    } else {
        selectedObjects = selectedObjects.filter(obj => obj !== object);
    }
    
    // Update visual state
    const item = e.target.closest('.object-list-item');
    if (item) {
        if (e.target.checked) {
            item.classList.add('selected');
        } else {
            item.classList.remove('selected');
        }
    }
}

// Select/deselect all objects
function selectAllObjects(selectAll) {
    if (selectAll) {
        selectedObjects = [...filteredObjects];
    } else {
        // Only remove objects that are currently in filtered list
        selectedObjects = selectedObjects.filter(obj => !filteredObjects.includes(obj));
    }
    
    updateObjectList();
}

// Clear all selected objects
function clearObjectSelection() {
    selectedObjects = [];
    updateSelectedObjectsDisplay();
    updateObjectList();
}

// Confirm selection and close dropdown
function confirmObjectSelection() {
    updateSelectedObjectsDisplay();
    closeObjectDropdown();
}

// Update the display of selected objects
function updateSelectedObjectsDisplay() {
    if (!selectedObjectsList || !selectedObjectsDisplay) return;
    
    selectedObjectsList.innerHTML = '';
    
    if (selectedObjects.length === 0) {
        selectedObjectsDisplay.style.display = 'none';
        updateSelectorButtonText();
        return;
    }
    
    selectedObjectsDisplay.style.display = 'block';
    
    selectedObjects.forEach(object => {
        const tag = document.createElement('div');
        tag.className = 'selected-object-tag';
        
        const text = document.createElement('span');
        text.textContent = object;
        
        const removeBtn = document.createElement('i');
        removeBtn.className = 'fas fa-times remove-tag';
        removeBtn.addEventListener('click', () => removeSelectedObject(object));
        
        tag.appendChild(text);
        tag.appendChild(removeBtn);
        selectedObjectsList.appendChild(tag);
    });
    
    updateSelectorButtonText();
}

// Remove individual selected object
function removeSelectedObject(object) {
    selectedObjects = selectedObjects.filter(obj => obj !== object);
    updateSelectedObjectsDisplay();
    updateObjectList();
}

// Update selector button text
function updateSelectorButtonText() {
    const selectorText = document.querySelector('.selector-text');
    if (!selectorText) return;
    
    if (selectedObjects.length === 0) {
        selectorText.textContent = 'Chọn objects...';
    } else if (selectedObjects.length === 1) {
        selectorText.textContent = `Đã chọn: ${selectedObjects[0]}`;
    } else {
        selectorText.textContent = `Đã chọn ${selectedObjects.length} objects`;
    }
}

// Lấy giá trị top-k từ ô nhập liệu
const topKInput = document.getElementById('topK');

// Xử lý tìm kiếm
function handleSearch(e) {
    e.preventDefault();
    
    const query = searchInput.value.trim();
    const ocrText = ocrInput ? ocrInput.value.trim() : '';
    const selectedModels = Array.from(document.querySelectorAll('input[name="models"]:checked')).map(checkbox => checkbox.value);
    const topK = topKInput ? parseInt(topKInput.value) : 100;
    
    // Validate input
    if (!query && !ocrText) {
        alert('Vui lòng nhập ít nhất một trong hai: từ khóa tìm kiếm hoặc OCR text');
        return;
    }
    
    if (isNaN(topK) || topK < 1) {
        alert('Giá trị Top K không hợp lệ');
        return;
    }
    
    if (selectedModels.length === 0) {
        alert('Vui lòng chọn ít nhất một model');
        return;
    }
    
    // Hiển thị loading
    if (loadingDiv) loadingDiv.style.display = 'flex';
    resultsDiv.innerHTML = '';
    
    // Tạo controller để có thể hủy request nếu cần
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000); // Timeout sau 30s
    
    // Tạo URL với tham số để tránh CORS preflight
    const url = new URL('http://localhost:5000/api/search');
    
    // Thêm tham số vào URL
    if (query) url.searchParams.append('query', query);
    if (ocrText) url.searchParams.append('ocr_text', ocrText);
    url.searchParams.append('models', JSON.stringify(selectedModels));
    if (selectedObjects.length > 0) {
        url.searchParams.append('objects', JSON.stringify(selectedObjects));
    }
    url.searchParams.append('topK', topK.toString());
    
    // Gọi API tìm kiếm với GET
    fetch(url.toString(), {
        method: 'GET',
        headers: {
            'Accept': 'application/json'
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
    .then(data => displayResults(data, { query, ocrText, selectedObjects }))
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
function displayResults(data, searchParams = {}) {
    if (!data || !data.paths || data.paths.length === 0) {
        const queryInfo = [];
        if (searchParams.query) queryInfo.push(`"${searchParams.query}"`);
        if (searchParams.ocrText) queryInfo.push(`OCR: "${searchParams.ocrText}"`);
        if (searchParams.selectedObjects && searchParams.selectedObjects.length > 0) {
            queryInfo.push(`Objects: ${searchParams.selectedObjects.join(', ')}`);
        }
        
        resultsDiv.innerHTML = `
            <div class="no-results" style="text-align: center; padding: 20px; color: #666;">
                <i class="fas fa-search" style="font-size: 2em; margin-bottom: 10px; opacity: 0.5;"></i>
                <p>Không tìm thấy kết quả phù hợp với ${queryInfo.join(' + ') || 'từ khóa tìm kiếm'}</p>
                ${searchParams.selectedObjects && searchParams.selectedObjects.length > 0 ? 
                    `<p style="font-size: 0.9em; color: #999;">Đã lọc theo objects: ${searchParams.selectedObjects.join(', ')}</p>` : ''
                }
            </div>
        `;
        return;
    }
    
    let html = `
        <div class="search-summary" style="margin-bottom: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #3498db;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                <i class="fas fa-info-circle" style="color: #3498db;"></i>
                <strong>Kết quả tìm kiếm:</strong>
            </div>
            <div style="font-size: 0.9rem; color: #6c757d;">
                ${searchParams.query ? `<div><strong>Query:</strong> "${searchParams.query}"</div>` : ''}
                ${searchParams.ocrText ? `<div><strong>OCR Text:</strong> "${searchParams.ocrText}"</div>` : ''}
                ${searchParams.selectedObjects && searchParams.selectedObjects.length > 0 ? 
                    `<div><strong>Objects:</strong> ${searchParams.selectedObjects.join(', ')}</div>` : ''
                }
                <div><strong>Tổng kết quả:</strong> ${data.paths.length} ảnh</div>
            </div>
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
            
        } catch (e) {
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
                        style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; cursor: pointer;"
                        onclick="openImageModal('${imagePath}', '${filename}', '${path}', '${score}')"
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

// Chức năng modal cho keyframe
function openImageModal(imagePath, filename, path, score) {
    const modal = document.getElementById('imageModal');
    const modalImage = document.getElementById('modalImage');
    const modalFilename = document.getElementById('modalFilename');
    const modalPath = document.getElementById('modalPath');
    const modalScore = document.getElementById('modalScore');
    
    // Set thông tin
    modalImage.src = imagePath;
    modalFilename.textContent = filename;
    modalPath.textContent = path;
    modalScore.textContent = score || 'Không có điểm số';
    
    // Hiển thị modal
    modal.style.display = 'block';
    document.body.style.overflow = 'hidden'; // Ngăn scroll body
}

function closeImageModal() {
    const modal = document.getElementById('imageModal');
    modal.style.display = 'none';
    document.body.style.overflow = 'auto'; // Khôi phục scroll body
}

// Event listeners cho modal
document.addEventListener('DOMContentLoaded', function() {
    const modal = document.getElementById('imageModal');
    const modalClose = document.querySelector('.modal-close');
    const modalOverlay = document.querySelector('.modal-overlay');
    
    // Đóng modal khi click nút X
    if (modalClose) {
        modalClose.addEventListener('click', closeImageModal);
    }
    
    // Đóng modal khi click vào overlay
    if (modalOverlay) {
        modalOverlay.addEventListener('click', closeImageModal);
    }
    
    // Đóng modal khi nhấn ESC
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && modal.style.display === 'block') {
            closeImageModal();
        }
    });
});

// Chạy ứng dụng khi DOM đã tải xong
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}
