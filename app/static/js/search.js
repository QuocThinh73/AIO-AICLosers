// ===================================================================
// SEARCH.JS - Module xử lý tìm kiếm và hiển thị kết quả
// ===================================================================

window.SearchModule = (function() {
    'use strict';
    
    // Private variables
    let isSearching = false;
    let currentController = null;
    
    // Private methods
    function validateSearchInput(query, ocrText, selectedModels, topK) {
        // Kiểm tra có ít nhất query hoặc OCR text
        if (!query && !ocrText) {
            alert('Vui lòng nhập ít nhất một trong hai: từ khóa tìm kiếm hoặc OCR text');
            return false;
        }
        
        // Kiểm tra TopK hợp lệ
        if (isNaN(topK) || topK < 1) {
            alert('Giá trị Top K không hợp lệ');
            return false;
        }
        
        // Kiểm tra có model được chọn
        if (selectedModels.length === 0) {
            alert('Vui lòng chọn ít nhất một model');
            return false;
        }
        
        return true;
    }
    
    function getSelectedModels() {
        return Array.from(document.querySelectorAll('input[name="models"]:checked'))
                   .map(checkbox => checkbox.value);
    }
    
    function buildSearchURL(query, ocrText, selectedModels, selectedObjects, topK) {
        const url = new URL('http://localhost:5000/api/search');
        
        if (query) url.searchParams.append('query', query);
        if (ocrText) url.searchParams.append('ocr_text', ocrText);
        url.searchParams.append('models', JSON.stringify(selectedModels));
        if (selectedObjects.length > 0) {
            url.searchParams.append('objects', JSON.stringify(selectedObjects));
        }
        url.searchParams.append('topK', topK.toString());
        
        return url.toString();
    }
    
    function setSearchingState(searching) {
        isSearching = searching;
        if (searching) {
            // Clear results when starting new search
            window.AppElements.resultsDiv.innerHTML = '';
        }
    }
    
    function handleSearchError(error) {
        let errorMessage = error.message || 'Có lỗi xảy ra khi tìm kiếm';
        
        if (error.name === 'AbortError') {
            errorMessage = 'Yêu cầu tìm kiếm đã hết thời gian chờ (30s). Vui lòng thử lại.';
        } else if (error.message.includes('Failed to fetch')) {
            errorMessage = 'Không thể kết nối đến máy chủ. Vui lòng kiểm tra kết nối mạng và thử lại.';
        }
        
        window.AppElements.resultsDiv.innerHTML = `
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
    }
    
    // Public methods
    function handleSearch(e) {
        e.preventDefault();
        
        if (isSearching) return;
        
        // Lấy dữ liệu từ form
        const query = window.AppElements.searchInput.value.trim();
        const ocrText = window.AppElements.ocrInput ? window.AppElements.ocrInput.value.trim() : '';
        const selectedModels = getSelectedModels();
        const selectedObjects = window.AppData.selectedObjects;
        const topK = window.AppElements.topKInput ? parseInt(window.AppElements.topKInput.value) : 100;
        
        // Validate input
        if (!validateSearchInput(query, ocrText, selectedModels, topK)) {
            return;
        }
        
        // Set searching state
        setSearchingState(true);
        
        // Hủy request trước đó nếu có
        if (currentController) {
            currentController.abort();
        }
        
        // Tạo controller mới
        currentController = new AbortController();
        const timeoutId = setTimeout(() => currentController.abort(), 30000);
        
        // Build URL
        const searchURL = buildSearchURL(query, ocrText, selectedModels, selectedObjects, topK);
        
        // Gửi API request
        fetch(searchURL, {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            },
            signal: currentController.signal
        })
        .then(async response => {
            clearTimeout(timeoutId);
            
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                const text = await response.text();
                throw new Error(`Server trả về lỗi không xác định (${response.status}): ${text.substring(0, 200)}`);
            }
            
            const data = await response.json();
            
            if (!response.ok) {
                const errorMessage = data.error || data.detail || data.message || `Lỗi HTTP: ${response.status}`;
                throw new Error(errorMessage);
            }
            
            return data;
        })
        .then(data => {
            displayResults(data, { query, ocrText, selectedObjects });
        })
        .catch(error => {
            if (error.name !== 'AbortError') {
                handleSearchError(error);
            }
        })
        .finally(() => {
            setSearchingState(false);
            currentController = null;
        });
    }
    
    function displayResults(data, searchParams = {}) {
        if (!data || !data.paths || data.paths.length === 0) {
            displayNoResults(searchParams);
            return;
        }
        
        const html = buildResultsHTML(data, searchParams);
        window.AppElements.resultsDiv.innerHTML = html;
    }
    
    function displayNoResults(searchParams) {
        const queryInfo = [];
        if (searchParams.query) queryInfo.push(`"${searchParams.query}"`);
        if (searchParams.ocrText) queryInfo.push(`OCR: "${searchParams.ocrText}"`);
        if (searchParams.selectedObjects && searchParams.selectedObjects.length > 0) {
            queryInfo.push(`Objects: ${searchParams.selectedObjects.join(', ')}`);
        }
        
        window.AppElements.resultsDiv.innerHTML = `
            <div class="no-results" style="text-align: center; padding: 20px; color: #666;">
                <i class="fas fa-search" style="font-size: 2em; margin-bottom: 10px; opacity: 0.5;"></i>
                <p>Không tìm thấy kết quả phù hợp với ${queryInfo.join(' + ') || 'từ khóa tìm kiếm'}</p>
                ${searchParams.selectedObjects && searchParams.selectedObjects.length > 0 ? 
                    `<p style="font-size: 0.9em; color: #999;">Đã lọc theo objects: ${searchParams.selectedObjects.join(', ')}</p>` : ''
                }
            </div>
        `;
    }
    
    function buildResultsHTML(data, searchParams) {
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
                
            const filename = data.filenames && data.filenames[index] 
                ? data.filenames[index] 
                : path.split('/').pop();
                
            let imagePath;
            try {
                const filename = path.split('/').pop();
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
                            onclick="window.ModalModule.openImageModal('${imagePath}', '${filename}', '${path}', '${score}')"
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
        return html;
    }
    
    // Public interface
    return {
        init: function() {
            if (window.AppElements.searchForm) {
                window.AppElements.searchForm.addEventListener('submit', handleSearch);
            }
        },
        
        handleSearch: handleSearch,
        displayResults: displayResults
    };
})(); 