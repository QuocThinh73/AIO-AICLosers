// ===================================================================
// OBJECTS.JS - Module for handling object selection
// ===================================================================

window.ObjectsModule = (function() {
    'use strict';
    
    // Private variables
    let filteredObjects = [];
    
    // DOM elements
    const elements = {
        objectSelectorBtn: document.getElementById('objectSelectorBtn'),
        objectDropdown: document.getElementById('objectDropdown'),
        objectSearch: document.getElementById('objectSearch'),
        objectList: document.getElementById('objectList'),
        selectedObjectsDisplay: document.getElementById('selectedObjectsDisplay'),
        selectedObjectsList: document.getElementById('selectedObjectsList'),
        clearSelectionBtn: document.getElementById('clearSelectionBtn'),
        selectAllObjectsBtn: document.getElementById('selectAllObjectsBtn'),
        clearAllObjectsBtn: document.getElementById('clearAllObjectsBtn'),
        confirmSelectionBtn: document.getElementById('confirmSelectionBtn')
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
                
                if (window.AppData.availableObjects.length === 0) {
                    showNoObjectsMessage();
                } else {
                    updateObjectList();
                }
            })
            .catch(error => {
                showObjectError(error.message);
            });
    }
    
    function setupEventListeners() {
        if (elements.objectSelectorBtn) {
            elements.objectSelectorBtn.addEventListener('click', toggleObjectDropdown);
        }
        
        if (elements.objectSearch) {
            elements.objectSearch.addEventListener('input', handleObjectSearch);
        }
        
        if (elements.clearSelectionBtn) {
            elements.clearSelectionBtn.addEventListener('click', clearObjectSelection);
        }
        
        if (elements.selectAllObjectsBtn) {
            elements.selectAllObjectsBtn.addEventListener('click', () => selectAllObjects(true));
        }
        
        if (elements.clearAllObjectsBtn) {
            elements.clearAllObjectsBtn.addEventListener('click', () => selectAllObjects(false));
        }
        
        if (elements.confirmSelectionBtn) {
            elements.confirmSelectionBtn.addEventListener('click', confirmObjectSelection);
        }
        
        document.addEventListener('click', (e) => {
            if (elements.objectDropdown && 
                !elements.objectDropdown.contains(e.target) && 
                !elements.objectSelectorBtn.contains(e.target)) {
                closeObjectDropdown();
            }
        });
    }
    
    function toggleObjectDropdown() {
        if (elements.objectDropdown) {
            const isVisible = elements.objectDropdown.style.display === 'flex';
            if (isVisible) {
                closeObjectDropdown();
            } else {
                openObjectDropdown();
            }
        }
    }
    
    function openObjectDropdown() {
        if (elements.objectDropdown && elements.objectSelectorBtn) {
            elements.objectDropdown.style.display = 'flex';
            elements.objectSelectorBtn.classList.add('active');
            if (elements.objectSearch) {
                elements.objectSearch.focus();
            }
        }
    }
    
    function closeObjectDropdown() {
        if (elements.objectDropdown && elements.objectSelectorBtn) {
            elements.objectDropdown.style.display = 'none';
            elements.objectSelectorBtn.classList.remove('active');
            if (elements.objectSearch) {
                elements.objectSearch.value = '';
                filteredObjects = [...window.AppData.availableObjects];
                updateObjectList();
            }
        }
    }
    
    function handleObjectSearch(e) {
        const searchTerm = e.target.value.toLowerCase().trim();
        
        if (searchTerm === '') {
            filteredObjects = [...window.AppData.availableObjects];
        } else {
            filteredObjects = window.AppData.availableObjects.filter(object => 
                object.toLowerCase().includes(searchTerm)
            );
        }
        
        updateObjectList();
    }
    
    function updateObjectList() {
        if (!elements.objectList) return;
        
        elements.objectList.innerHTML = '';
        
        if (filteredObjects.length === 0) {
            elements.objectList.innerHTML = `
                <div class="no-objects-message">
                    No objects found
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
            checkbox.checked = window.AppData.selectedObjects.includes(object);
            checkbox.addEventListener('change', (e) => handleObjectSelection(e, object));
            
            const label = document.createElement('label');
            label.textContent = object;
            label.addEventListener('click', () => {
                checkbox.checked = !checkbox.checked;
                handleObjectSelection({ target: checkbox }, object);
            });
            
            item.appendChild(checkbox);
            item.appendChild(label);
            
            if (window.AppData.selectedObjects.includes(object)) {
                item.classList.add('selected');
            }
            
            elements.objectList.appendChild(item);
        });
    }
    
    function handleObjectSelection(e, object) {
        if (e.target.checked) {
            if (!window.AppData.selectedObjects.includes(object)) {
                window.AppData.selectedObjects.push(object);
            }
        } else {
            window.AppData.selectedObjects = window.AppData.selectedObjects.filter(obj => obj !== object);
        }
        
        const item = e.target.closest('.object-list-item');
        if (item) {
            if (e.target.checked) {
                item.classList.add('selected');
            } else {
                item.classList.remove('selected');
            }
        }
    }
    
    function selectAllObjects(selectAll) {
        if (selectAll) {
            filteredObjects.forEach(object => {
                if (!window.AppData.selectedObjects.includes(object)) {
                    window.AppData.selectedObjects.push(object);
                }
            });
        } else {
            window.AppData.selectedObjects = window.AppData.selectedObjects.filter(obj => 
                !filteredObjects.includes(obj)
            );
        }
        
        updateObjectList();
    }
    
    function clearObjectSelection() {
        window.AppData.selectedObjects = [];
        updateSelectedObjectsDisplay();
        updateObjectList();
    }
    
    function confirmObjectSelection() {
        updateSelectedObjectsDisplay();
        closeObjectDropdown();
    }
    
    function updateSelectedObjectsDisplay() {
        if (!elements.selectedObjectsList || !elements.selectedObjectsDisplay) return;
        
        elements.selectedObjectsList.innerHTML = '';
        
        if (window.AppData.selectedObjects.length === 0) {
            elements.selectedObjectsDisplay.style.display = 'none';
            updateSelectorButtonText();
            return;
        }
        
        elements.selectedObjectsDisplay.style.display = 'block';
        
        window.AppData.selectedObjects.forEach(object => {
            const tag = document.createElement('div');
            tag.className = 'selected-object-tag';
            
            const text = document.createElement('span');
            text.textContent = object;
            
            const removeBtn = document.createElement('i');
            removeBtn.className = 'fas fa-times remove-tag';
            removeBtn.addEventListener('click', () => removeSelectedObject(object));
            
            tag.appendChild(text);
            tag.appendChild(removeBtn);
            elements.selectedObjectsList.appendChild(tag);
        });
        
        updateSelectorButtonText();
    }
    
    function removeSelectedObject(object) {
        window.AppData.selectedObjects = window.AppData.selectedObjects.filter(obj => obj !== object);
        updateSelectedObjectsDisplay();
        updateObjectList();
    }
    
    function updateSelectorButtonText() {
        const selectorText = document.querySelector('.selector-text');
        if (!selectorText) return;
        
        if (window.AppData.selectedObjects.length === 0) {
            selectorText.textContent = 'Select objects...';
        } else if (window.AppData.selectedObjects.length === 1) {
            selectorText.textContent = `Selected: ${window.AppData.selectedObjects[0]}`;
        } else {
            selectorText.textContent = `Selected ${window.AppData.selectedObjects.length} objects`;
        }
    }
    
    function showNoObjectsMessage() {
        if (elements.objectList) {
            elements.objectList.innerHTML = `
                <div class="no-objects-message">
                    No objects available
                </div>
            `;
        }
    }
    
    function showObjectError(errorMessage) {
        if (elements.objectList) {
            elements.objectList.innerHTML = `
                <div class="error-message" style="color: #721c24; padding: 10px; background: #f8d7da; border-radius: 4px; font-size: 0.9rem;">
                    <i class="fas fa-exclamation-triangle"></i> Unable to load objects list: ${errorMessage}
                </div>
            `;
        }
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
        
        clearSelection: clearObjectSelection,
        reloadObjects: loadObjects,
        updateDisplay: updateSelectedObjectsDisplay
    };
})(); 