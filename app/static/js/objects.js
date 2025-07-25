// ===================================================================
// OBJECTS.JS - Module for handling object selection (DISABLED)
// Object filtering functionality has been removed from the pipeline
// ===================================================================

window.ObjectsModule = (function() {
    'use strict';
    
    // Private variables
    let filteredObjects = [];
    let currentHighlighted = -1;
    let isDropdownVisible = false;
    
    // DOM elements
    const elements = {
        objectInput: document.getElementById('objectInput'),
        objectDropdown: document.getElementById('objectDropdown'),
        selectedObjectsTags: document.getElementById('selectedObjectsTags')
    };
    
    // Private methods
    function loadObjects() {
        fetch('/api/objects')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                window.AppData.availableObjects = data.objects || data || [];
                filteredObjects = [...window.AppData.availableObjects];
            })
            .catch(error => {
                console.error('Error loading objects:', error);
            });
    }
    
    function setupEventListeners() {
        if (elements.objectInput) {
            elements.objectInput.addEventListener('input', handleInputChange);
            elements.objectInput.addEventListener('keydown', handleKeyDown);
            elements.objectInput.addEventListener('focus', handleFocus);
            elements.objectInput.addEventListener('blur', hideDropdownDelayed);
        }
        
        // Close dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (elements.objectDropdown && 
                !elements.objectInput.contains(e.target) && 
                !elements.objectDropdown.contains(e.target)) {
                hideDropdown();
            }
        });
    }
    
    function handleInputChange(e) {
        const value = e.target.value.toLowerCase().trim();
        
        if (value === '') {
            // Show all objects when input is empty
            filteredObjects = window.AppData.availableObjects.filter(object => 
                !window.AppData.selectedObjects.includes(object)
            );
        } else {
            // Filter objects based on input
            filteredObjects = window.AppData.availableObjects.filter(object => 
                object.toLowerCase().includes(value) && 
                !window.AppData.selectedObjects.includes(object)
            );
        }
        
        currentHighlighted = -1;
        updateDropdown();
        showDropdown();
    }
    
    function handleKeyDown(e) {
        if (!isDropdownVisible) {
            return;
        }
        
        const items = elements.objectDropdown.querySelectorAll('.object-item:not(.no-select)');
        
        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                currentHighlighted = Math.min(currentHighlighted + 1, items.length - 1);
                updateHighlight(items);
                scrollToHighlighted();
                break;
                
            case 'ArrowUp':
                e.preventDefault();
                currentHighlighted = Math.max(currentHighlighted - 1, -1);
                updateHighlight(items);
                scrollToHighlighted();
                break;
                
            case 'Enter':
                e.preventDefault();
                if (currentHighlighted >= 0 && items[currentHighlighted]) {
                    selectObject(items[currentHighlighted].textContent);
                }
                // Only allow selecting from the available list, no custom objects
                break;
                
            case 'Escape':
                hideDropdown();
                elements.objectInput.blur();
                break;
                
            case ',':
            case 'Tab':
                e.preventDefault();
                // Only select if there's a highlighted item, no custom objects
                if (currentHighlighted >= 0 && items[currentHighlighted]) {
                    selectObject(items[currentHighlighted].textContent);
                }
                break;
        }
    }
    
    function updateHighlight(items) {
        items.forEach((item, index) => {
            if (index === currentHighlighted) {
                item.classList.add('highlighted');
            } else {
                item.classList.remove('highlighted');
            }
        });
    }
    
    function scrollToHighlighted() {
        if (currentHighlighted >= 0) {
            const items = elements.objectDropdown.querySelectorAll('.object-item:not(.no-select)');
            if (items[currentHighlighted]) {
                items[currentHighlighted].scrollIntoView({
                    block: 'nearest',
                    behavior: 'smooth'
                });
            }
        }
    }
    
    function handleFocus() {
        // Show all available objects when focusing (excluding selected ones)
        filteredObjects = window.AppData.availableObjects.filter(object => 
            !window.AppData.selectedObjects.includes(object)
        );
        currentHighlighted = -1;
        updateDropdown();
        showDropdown();
    }
    
    function showDropdown() {
        if (elements.objectDropdown) {
            elements.objectDropdown.style.display = 'block';
            isDropdownVisible = true;
        }
    }
    
    function hideDropdown() {
        if (elements.objectDropdown) {
            elements.objectDropdown.style.display = 'none';
            isDropdownVisible = false;
            currentHighlighted = -1;
        }
    }
    
    function hideDropdownDelayed() {
        // Delay hiding to allow clicks on dropdown items
        setTimeout(() => {
            hideDropdown();
        }, 150);
    }
    
    function updateDropdown() {
        if (!elements.objectDropdown) return;
        
        elements.objectDropdown.innerHTML = '';
        
        if (filteredObjects.length === 0) {
            const noItems = document.createElement('div');
            noItems.className = 'object-item no-select';
            if (elements.objectInput.value.trim()) {
                noItems.textContent = `Không tìm thấy object "${elements.objectInput.value.trim()}"`;
            } else {
                noItems.textContent = 'Không có object nào khả dụng';
            }
            noItems.style.fontStyle = 'italic';
            noItems.style.color = '#6c757d';
            noItems.style.cursor = 'default';
            elements.objectDropdown.appendChild(noItems);
            return;
        }
        
        filteredObjects.forEach(object => {
            const item = document.createElement('div');
            item.className = 'object-item';
            item.textContent = object;
            
            if (window.AppData.selectedObjects.includes(object)) {
                item.classList.add('selected');
            }
            
            item.addEventListener('mousedown', (e) => {
                e.preventDefault(); // Prevent blur event
                selectObject(object);
            });
            
            elements.objectDropdown.appendChild(item);
        });
    }
    
    function selectObject(objectName) {
        if (!objectName || 
            window.AppData.selectedObjects.includes(objectName) ||
            !window.AppData.availableObjects.includes(objectName)) {
            return;
        }
        
        window.AppData.selectedObjects.push(objectName);
        elements.objectInput.value = '';
        hideDropdown();
        updateTagsDisplay();
        elements.objectInput.focus();
    }
    
    function removeSelectedObject(object) {
        window.AppData.selectedObjects = window.AppData.selectedObjects.filter(obj => obj !== object);
        updateTagsDisplay();
        
        // Update dropdown if it's visible
        if (isDropdownVisible) {
            handleFocus(); // Refresh the dropdown with updated selections
        }
    }
    
    function updateTagsDisplay() {
        if (!elements.selectedObjectsTags) return;
        
        elements.selectedObjectsTags.innerHTML = '';
        
        window.AppData.selectedObjects.forEach(object => {
            const tag = document.createElement('div');
            tag.className = 'object-tag';
            
            const text = document.createElement('span');
            text.textContent = object;
            
            const removeBtn = document.createElement('i');
            removeBtn.className = 'fas fa-times remove-tag';
            removeBtn.addEventListener('click', () => removeSelectedObject(object));
            
            tag.appendChild(text);
            tag.appendChild(removeBtn);
            elements.selectedObjectsTags.appendChild(tag);
        });
    }
    
    // Public interface
    return {
        init: function() {
            setupEventListeners();
            loadObjects();
        },
        
        getSelectedObjects: function() {
            return window.AppData.selectedObjects;
        },
        
        clearSelection: function() {
            window.AppData.selectedObjects = [];
            updateTagsDisplay();
        },
        
        reloadObjects: loadObjects,
        updateDisplay: updateTagsDisplay
    };
})(); 