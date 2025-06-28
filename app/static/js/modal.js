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
    
    function navigateToPrevKeyframe() {
        if (currentKeyframeIndex > 0) {
            currentKeyframeIndex--;
            loadKeyframeAtIndex(currentKeyframeIndex);
        }
    }
    
    function navigateToNextKeyframe() {
        if (currentKeyframeIndex < currentKeyframes.length - 1) {
            currentKeyframeIndex++;
            loadKeyframeAtIndex(currentKeyframeIndex);
        }
    }
    
    function loadKeyframeAtIndex(index) {
        if (index < 0 || index >= currentKeyframes.length) return;
        
        const keyframe = currentKeyframes[index];
        const imagePath = keyframe.path;
        const filename = keyframe.filename;
        
        // Update modal with new keyframe data
        loadModalContent(imagePath, filename, 'Keyframe từ navigation', currentScore);
        updateNavigationInfo();
        updateNavigationButtons();
    }
    
    function updateNavigationInfo() {
        if (elements.modalKeyframeInfo && currentKeyframes.length > 0) {
            elements.modalKeyframeInfo.textContent = 
                `Keyframe ${currentKeyframeIndex + 1} / ${currentKeyframes.length}`;
        }
    }
    
    function updateNavigationButtons() {
        if (elements.modalPrevBtn) {
            elements.modalPrevBtn.disabled = currentKeyframeIndex <= 0;
        }
        
        if (elements.modalNextBtn) {
            elements.modalNextBtn.disabled = currentKeyframeIndex >= currentKeyframes.length - 1;
        }
    }
    
    function loadModalContent(imagePath, filename, path, score) {
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
    }
    
    function openImageModal(imagePath, filename, path, score) {
        if (!elements.modal || !elements.modalImage) {
            return;
        }
        
        // Store current score for navigation
        currentScore = score;
        
        // Load keyframes for the same video
        fetch(`/api/video-keyframes/${filename}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // Store keyframes and find current index
                currentKeyframes = data.keyframes;
                currentKeyframeIndex = currentKeyframes.findIndex(kf => kf.filename === filename);
                
                if (currentKeyframeIndex === -1) {
                    // If not found, add current keyframe and set as index 0
                    currentKeyframes.unshift({
                        filename: filename,
                        path: imagePath,
                        frame_number: 0
                    });
                    currentKeyframeIndex = 0;
                }
                
                // Update navigation info and buttons
                updateNavigationInfo();
                updateNavigationButtons();
                
            })
            .catch(error => {
                console.error('Error loading keyframes:', error);
                // Fallback: single keyframe mode
                currentKeyframes = [{
                    filename: filename,
                    path: imagePath,
                    frame_number: 0
                }];
                currentKeyframeIndex = 0;
                updateNavigationInfo();
                updateNavigationButtons();
            });
        
        // Load modal content for current keyframe
        loadModalContent(imagePath, filename, path, score);
        
        // Show modal
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
        
        // Reset navigation state
        currentKeyframes = [];
        currentKeyframeIndex = -1;
        currentScore = null;
        
        // Reset navigation info
        if (elements.modalKeyframeInfo) {
            elements.modalKeyframeInfo.textContent = '-';
        }
        
        // Reset navigation buttons
        if (elements.modalPrevBtn) {
            elements.modalPrevBtn.disabled = true;
        }
        if (elements.modalNextBtn) {
            elements.modalNextBtn.disabled = true;
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