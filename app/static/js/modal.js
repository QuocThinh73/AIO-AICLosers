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
        
        elements.modalImage.src = imagePath;
        
        if (elements.modalFilename) {
            elements.modalFilename.textContent = filename || 'Không có tên file';
        }
        
        if (elements.modalPath) {
            elements.modalPath.textContent = path || 'Không có đường dẫn';
        }
        
        if (elements.modalScore) {
            elements.modalScore.textContent = score || 'Không có điểm số';
        }
        
        elements.modal.style.display = 'block';
        document.body.style.overflow = 'hidden';
        
        elements.modalImage.onerror = function() {
            this.src = '/static/images/no-image.jpg';
        };
    }
    
    function closeImageModal() {
        if (!elements.modal) return;
        
        elements.modal.style.display = 'none';
        document.body.style.overflow = 'auto';
        
        if (elements.modalImage) {
            elements.modalImage.src = '';
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