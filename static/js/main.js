document.addEventListener('DOMContentLoaded', function() {
    const searchForm = document.getElementById('searchForm');
    const searchInput = document.getElementById('searchInput');
    const modelSelect = document.getElementById('modelSelect');
    const resultsContainer = document.getElementById('results');
    const loadingIndicator = document.querySelector('.loading');

    // Xử lý sự kiện submit form
    searchForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const query = searchInput.value.trim();
        const model = modelSelect.value;
        
        if (!query) {
            alert('Vui lòng nhập từ khóa tìm kiếm');
            return;
        }
        
        // Hiển thị loading
        showLoading(true);
        resultsContainer.innerHTML = '';
        
        // Gọi API tìm kiếm
        searchImages(query, model);
    });

    // Hàm gọi API tìm kiếm
    async function searchImages(query, model) {
        try {
            showLoading(true);
            resultsContainer.innerHTML = '';
            
            console.log('Đang gửi yêu cầu tìm kiếm...', { query, model });
            
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    top_k: 12,  // Lấy 12 kết quả
                    model_name: model
                })
            });

            const data = await response.json();
            console.log('Phản hồi từ API:', data);

            if (!response.ok) {
                let errorMsg = `Lỗi ${response.status}: ${data.error || 'Lỗi không xác định'}`;
                if (data.details) {
                    errorMsg += `\n\nChi tiết: ${data.details}`;
                }
                throw new Error(errorMsg);
            }

            if (!data.paths || data.paths.length === 0) {
                throw new Error('Không tìm thấy kết quả nào phù hợp.');
            }

            displayResults(data);
        } catch (error) {
            console.error('Lỗi khi tìm kiếm:', error);
            resultsContainer.innerHTML = `
                <div class="error-message" style="color: #d32f2f; background: #ffebee; padding: 15px; border-radius: 4px; margin: 10px 0;">
                    <h3>Đã xảy ra lỗi khi tìm kiếm</h3>
                    <p>${error.message}</p>
                    <p>Vui lòng thử lại với từ khóa khác hoặc liên hệ quản trị viên nếu lỗi vẫn tiếp diễn.</p>
                </div>`;
        } finally {
            showLoading(false);
        }
    }

    // Hiển thị kết quả tìm kiếm
    function displayResults(data) {
        if (!data.paths || data.paths.length === 0) {
            resultsContainer.innerHTML = '<p>Không tìm thấy kết quả nào phù hợp.</p>';
            return;
        }

        let html = '<div class="results-grid">';
        
        data.paths.forEach((path, index) => {
            const score = data.scores ? (data.scores[index] * 100).toFixed(2) : '0';
            const filename = path.split('/').pop();
            
            // Tạo đường dẫn ảnh đúng định dạng
            let imgSrc;
            if (path.startsWith('http')) {
                imgSrc = path;  // Nếu là URL đầy đủ
            } else if (path.includes('keyframes')) {
                // Nếu đường dẫn chứa 'keyframes', sử dụng route /data/
                const relPath = path.replace(/^.*?keyframes[\\/]?/i, '');
                imgSrc = `/data/keyframes/${relPath}`;
            } else {
                // Mặc định sử dụng đường dẫn tương đối
                imgSrc = `/${path}`.replace(/\\/g, '/');
            }
            
            console.log(`Image ${index + 1}:`, { path, imgSrc }); // Log để debug
            
            html += `
                <div class="result-item">
                    <img src="${imgSrc}" 
                         alt="Kết quả tìm kiếm" 
                         class="result-image" 
                         onerror="this.onerror=null; this.src='/static/images/no-image.png'"
                         loading="lazy">
                    <div class="result-info">
                        ${score ? `<span class="result-score">Độ phù hợp: ${score}%</span>` : ''}
                        <p>${filename}</p>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        resultsContainer.innerHTML = html;
    }

    // Hiển thị/Ẩn loading
    function showLoading(show) {
        if (show) {
            loadingIndicator.style.display = 'block';
        } else {
            loadingIndicator.style.display = 'none';
        }
    }
});
