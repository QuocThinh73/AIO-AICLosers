// ===================================================================
// MODAL.JS - Module for handling modal display of image details
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
        modalKeyframeInfo: document.getElementById('modalKeyframeInfo'),
        modalClose: document.querySelector('.modal-close'),
        modalOverlay: document.querySelector('.modal-overlay'),
        modalPrevBtn: document.getElementById('modalPrevBtn'),
        modalNextBtn: document.getElementById('modalNextBtn')
    };
    
    // State variables
    let currentKeyframes = [];
    let currentKeyframeIndex = -1;
    let currentScore = null;
    
    // Private methods
    function setupEventListeners() {
        if (elements.modalClose) {
            elements.modalClose.addEventListener('click', closeImageModal);
        }
        
        if (elements.modalOverlay) {
            elements.modalOverlay.addEventListener('click', closeImageModal);
        }
        
        if (elements.modalPrevBtn) {
            elements.modalPrevBtn.addEventListener('click', navigateToPrevKeyframe);
        }
        
        if (elements.modalNextBtn) {
            elements.modalNextBtn.addEventListener('click', navigateToNextKeyframe);
        }
        
        document.addEventListener('keydown', function(e) {
            if (elements.modal && elements.modal.style.display === 'block') {
                switch(e.key) {
                    case 'Escape':
                        closeImageModal();
                        break;
                    case 'ArrowLeft':
                        e.preventDefault();
                        navigateToPrevKeyframe();
                        break;
                    case 'ArrowRight':
                        e.preventDefault();
                        navigateToNextKeyframe();
                        break;
                }
            }
        });
    }
    
    function openImageModal(imagePath, filename, path, score) {
        if (!elements.modal || !elements.modalImage) {
            return;
        }
        
        // Get video information from API
        fetch(`/api/video-info/${filename}`)
            .then(response => response.json())
            .then(videoInfo => {
                if (videoInfo.error) {
                    throw new Error(videoInfo.error);
                }
                
                // Replace image with video player
                elements.modalImage.style.display = 'none';
                
                // Create video element if not exists
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
        
        // Display detailed information
        if (elements.modalFilename) {
            elements.modalFilename.textContent = filename || 'No filename';
        }
        
        if (elements.modalPath && elements.modalPath.parentElement) {
            elements.modalPath.parentElement.style.display = 'block';
            // Path will be updated in API call or use default
            if (!elements.modalPath.textContent || elements.modalPath.textContent === 'No path') {
                elements.modalPath.textContent = path || 'No path';
            }
        }
        
        if (elements.modalScore && elements.modalScore.parentElement) {
            elements.modalScore.parentElement.style.display = 'block';
            elements.modalScore.textContent = score || 'No score';
        }
        
        elements.modal.style.display = 'block';
        document.body.style.overflow = 'hidden';
        
        // Set onerror handler only if modal is visible and image is displayed
        elements.modalImage.onerror = function() {
            if (elements.modal && elements.modal.style.display === 'block') {
                this.src = '/static/images/no-image.png';
            }
        };
    }
    
    function closeImageModal() {
        if (!elements.modal) return;
        
        elements.modal.style.display = 'none';
        document.body.style.overflow = 'auto';
        
        if (elements.modalImage) {
            // Remove onerror handler before clearing src to prevent infinite requests
            elements.modalImage.onerror = null;
            elements.modalImage.src = '';
            elements.modalImage.style.display = 'block';
        }
        
        // Clean up video player
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