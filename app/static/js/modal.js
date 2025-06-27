// ===================================================================
// MODAL.JS - Module xử lý modal hiển thị ảnh chi tiết
// ===================================================================

window.ModalModule = (function() {
    'use strict';
    
    // DOM elements
    const elements = {
        modal: document.getElementById('imageModal'),
        modalImage: document.getElementById('modalImage'),
        modalFilename: document.getElementById('modalFilename'),
        modalPath: document.getElementById('modalPath'),
        modalScore: document.getElementById('modalScore'),
        modalClose: document.querySelector('.modal-close'),
        modalOverlay: document.querySelector('.modal-overlay')
    };
    
    // Private methods
    function setupEventListeners() {
        if (elements.modalClose) {
            elements.modalClose.addEventListener('click', closeImageModal);
        }
        
        if (elements.modalOverlay) {
            elements.modalOverlay.addEventListener('click', closeImageModal);
        }
        
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && elements.modal && elements.modal.style.display === 'block') {
                closeImageModal();
            }
        });
    }
    
    function openImageModal(imagePath, filename, path, score) {
        if (!elements.modal || !elements.modalImage) {
            return;
        }
        
        // Lấy thông tin video từ API
        fetch(`/api/video-info/${filename}`)
            .then(response => response.json())
            .then(videoInfo => {
                if (videoInfo.error) {
                    throw new Error(videoInfo.error);
                }
                
                // Thay ảnh bằng video player
                elements.modalImage.style.display = 'none';
                
                // Tạo video element nếu chưa có
                let videoElement = document.getElementById('modalVideo');
                if (!videoElement) {
                    videoElement = document.createElement('video');
                    videoElement.id = 'modalVideo';
                    videoElement.controls = true;
                    videoElement.style.maxWidth = '100%';
                    videoElement.style.maxHeight = '70vh';
                    videoElement.style.objectFit = 'contain';
                    elements.modalImage.parentElement.appendChild(videoElement);
                }
                
                videoElement.style.display = 'block';
                videoElement.src = videoInfo.video_path;
                
                // Seek to specific timestamp when video loads
                videoElement.addEventListener('loadeddata', function() {
                    this.currentTime = videoInfo.timestamp;
                }, { once: true });
                
                // Update modal info with video details
                if (elements.modalPath) {
                    elements.modalPath.textContent = `Video: ${videoInfo.video_path} | Frame: ${videoInfo.frame_number} | Time: ${videoInfo.timestamp.toFixed(2)}s | FPS: ${videoInfo.fps.toFixed(1)}`;
                }
                
            })
            .catch(error => {
                console.error('Error loading video info:', error);
                // Fallback to image if API fails
                elements.modalImage.style.display = 'block';
                elements.modalImage.src = imagePath;
            });
        
        // Hiển thị thông tin chi tiết
        if (elements.modalFilename) {
            elements.modalFilename.textContent = filename || 'Không có tên file';
        }
        
        if (elements.modalPath && elements.modalPath.parentElement) {
            elements.modalPath.parentElement.style.display = 'block';
            // Path sẽ được cập nhật trong API call hoặc dùng default
            if (!elements.modalPath.textContent || elements.modalPath.textContent === 'Không có đường dẫn') {
                elements.modalPath.textContent = path || 'Không có đường dẫn';
            }
        }
        
        if (elements.modalScore && elements.modalScore.parentElement) {
            elements.modalScore.parentElement.style.display = 'block';
            elements.modalScore.textContent = score || 'Không có điểm số';
        }
        
        elements.modal.style.display = 'block';
        document.body.style.overflow = 'hidden';
        
        elements.modalImage.onerror = function() {
            this.src = '/static/images/no-image.png';
        };
    }
    
    function closeImageModal() {
        if (!elements.modal) return;
        
        elements.modal.style.display = 'none';
        document.body.style.overflow = 'auto';
        
        if (elements.modalImage) {
            elements.modalImage.src = '';
            elements.modalImage.style.display = 'block';
        }
        
        // Dọn dẹp video player
        const videoElement = document.getElementById('modalVideo');
        if (videoElement) {
            videoElement.pause();
            videoElement.src = '';
            videoElement.style.display = 'none';
        }
    }
    
    // Public interface
    return {
        init: function() {
            setupEventListeners();
        },
        
        openImageModal: openImageModal,
        closeImageModal: closeImageModal,
        
        open: openImageModal,
        close: closeImageModal
    };
})(); 