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
        modalFrameNumber: document.getElementById('modalFrameNumber'),
        modalFramePosition: document.getElementById('modalFramePosition'),
        modalClose: document.querySelector('.modal-close'),
        modalOverlay: document.querySelector('.modal-overlay'),
        modalPrevBtn: document.getElementById('modalPrevBtn'),
        modalNextBtn: document.getElementById('modalNextBtn'),
        keyframeThumbnails: document.getElementById('keyframeThumbnails')
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
        renderKeyframeThumbnails();
        
        // Update filename display
        if (elements.modalFilename) {
            elements.modalFilename.textContent = filename || 'N/A';
        }
    }
    
    function updateNavigationInfo() {
        if (elements.modalFramePosition && currentKeyframes.length > 0) {
            elements.modalFramePosition.textContent = 
                `${currentKeyframeIndex + 1} / ${currentKeyframes.length}`;
        }
    }
    
    function updateFrameNumber(filename) {
        if (elements.modalFrameNumber) {
            // Extract frame number from filename: L01_V003_015190.jpg -> 015190
            const parts = filename.split('_');
            if (parts.length >= 3) {
                const frameStr = parts[2].replace('.jpg', '');
                elements.modalFrameNumber.textContent = frameStr;
            }
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
    
    function renderKeyframeThumbnails() {
        if (!elements.keyframeThumbnails || currentKeyframes.length === 0) return;
        
        // Clear existing thumbnails
        elements.keyframeThumbnails.innerHTML = '';
        
        // Calculate range: 2 before, current, 2 after
        const startIndex = Math.max(0, currentKeyframeIndex - 2);
        const endIndex = Math.min(currentKeyframes.length - 1, currentKeyframeIndex + 2);
        
        // Create thumbnails for the range
        for (let i = startIndex; i <= endIndex; i++) {
            const keyframe = currentKeyframes[i];
            const thumbnailDiv = document.createElement('div');
            thumbnailDiv.className = 'keyframe-thumbnail';
            
            // Mark current keyframe
            if (i === currentKeyframeIndex) {
                thumbnailDiv.classList.add('current');
            }
            
            // Add click handler
            thumbnailDiv.addEventListener('click', () => handleThumbnailClick(i));
            
            // Create image element
            const img = document.createElement('img');
            img.src = keyframe.path;
            img.alt = `Keyframe ${keyframe.frame_number}`;
            
            // Handle image loading errors
            img.onerror = function() {
                this.src = '/static/images/no-image.png';
            };
            
            // Create frame indicator
            const indicator = document.createElement('div');
            indicator.className = 'frame-indicator';
            indicator.textContent = `${i + 1}`;
            
            // Append elements
            thumbnailDiv.appendChild(img);
            thumbnailDiv.appendChild(indicator);
            elements.keyframeThumbnails.appendChild(thumbnailDiv);
        }
        
        // Add disabled placeholders if we have less than 5 frames
        const totalThumbnails = endIndex - startIndex + 1;
        if (totalThumbnails < 5) {
            const missingCount = 5 - totalThumbnails;
            
            // Add placeholders at the beginning if needed
            if (startIndex === 0 && currentKeyframeIndex < 2) {
                for (let i = 0; i < Math.min(missingCount, 2 - currentKeyframeIndex); i++) {
                    const placeholder = createPlaceholderThumbnail();
                    elements.keyframeThumbnails.insertBefore(placeholder, elements.keyframeThumbnails.firstChild);
                }
            }
            
            // Add placeholders at the end if needed
            if (endIndex === currentKeyframes.length - 1) {
                const remainingPlaceholders = 5 - elements.keyframeThumbnails.children.length;
                for (let i = 0; i < remainingPlaceholders; i++) {
                    const placeholder = createPlaceholderThumbnail();
                    elements.keyframeThumbnails.appendChild(placeholder);
                }
            }
        }
    }
    
    function createPlaceholderThumbnail() {
        const placeholder = document.createElement('div');
        placeholder.className = 'keyframe-thumbnail disabled';
        
        const indicator = document.createElement('div');
        indicator.className = 'frame-indicator';
        indicator.textContent = '-';
        
        placeholder.appendChild(indicator);
        return placeholder;
    }
    
    function handleThumbnailClick(index) {
        if (index >= 0 && index < currentKeyframes.length && index !== currentKeyframeIndex) {
            currentKeyframeIndex = index;
            loadKeyframeAtIndex(currentKeyframeIndex);
        }
    }
    
    function updateThumbnailHighlight() {
        if (!elements.keyframeThumbnails) return;
        
        // Remove current class from all thumbnails
        const thumbnails = elements.keyframeThumbnails.querySelectorAll('.keyframe-thumbnail');
        thumbnails.forEach(thumb => thumb.classList.remove('current'));
        
        // Re-render thumbnails to update the selection
        renderKeyframeThumbnails();
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
                    videoElement.style.maxHeight = '50vh';
                    videoElement.style.objectFit = 'contain';
                    elements.modalImage.parentElement.appendChild(videoElement);
                }
                
                videoElement.style.display = 'block';
                videoElement.src = videoInfo.video_path;
                
                // Seek to specific timestamp when video loads
                videoElement.addEventListener('loadeddata', function() {
                    this.currentTime = videoInfo.timestamp;
                }, { once: true });
                
            })
            .catch(error => {
                console.error('Error loading video info:', error);
                // Fallback to image if API fails
                elements.modalImage.style.display = 'block';
                elements.modalImage.src = imagePath;
            });
        
        // Display filename and frame info
        if (elements.modalFilename) {
            elements.modalFilename.textContent = filename || 'N/A';
        }
        
        // Update frame number
        updateFrameNumber(filename);
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
                renderKeyframeThumbnails();
                
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
                renderKeyframeThumbnails();
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
        
        // Reset frame info
        if (elements.modalFrameNumber) {
            elements.modalFrameNumber.textContent = '-';
        }
        if (elements.modalFramePosition) {
            elements.modalFramePosition.textContent = '-';
        }
        
        // Reset navigation buttons
        if (elements.modalPrevBtn) {
            elements.modalPrevBtn.disabled = true;
        }
        if (elements.modalNextBtn) {
            elements.modalNextBtn.disabled = true;
        }
        
        // Clear thumbnails
        if (elements.keyframeThumbnails) {
            elements.keyframeThumbnails.innerHTML = '';
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