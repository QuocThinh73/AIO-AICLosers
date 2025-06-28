// ===================================================================
// MODELS.JS - Module for handling model selection
// ===================================================================

window.ModelsModule = (function() {
    'use strict';
    
    // Private methods
    function loadModels() {
        fetch('/api/models')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                window.AppData.availableModels = data.models || [];
                
                if (window.AppData.availableModels.length > 0) {
                    updateModelSelect();
                }
            })
            .catch(error => {
                showModelError(error.message);
            });
    }
    
    function updateModelSelect() {
        const modelSelect = window.AppElements.modelSelect;
        modelSelect.innerHTML = ''; 
        
        window.AppData.availableModels.forEach(model => {
            const chip = document.createElement('div');
            chip.className = 'model-chip selected'; // All selected by default
            chip.dataset.model = model;
            
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.value = model;
            checkbox.name = 'models';
            checkbox.checked = true;

            const span = document.createElement('span');
            span.textContent = model;

            chip.appendChild(checkbox);
            chip.appendChild(span);
            
            // Click handler for chip
            chip.addEventListener('click', function() {
                checkbox.checked = !checkbox.checked;
                if (checkbox.checked) {
                    chip.classList.add('selected');
                } else {
                    chip.classList.remove('selected');
                }
            });
            
            modelSelect.appendChild(chip);
        });
    }
    
    function showModelError(errorMessage) {
        window.AppElements.modelSelect.innerHTML = `
            <div class="error-message" style="color: #721c24; padding: 10px; background: #f8d7da; border-radius: 4px; font-size: 0.9rem;">
                <i class="fas fa-exclamation-triangle"></i> Unable to load models list: ${errorMessage}
            </div>
        `;
    }
    
    // Public interface
    return {
        init: function() {
            loadModels();
        },
        
        getSelectedModels: function() {
            return Array.from(document.querySelectorAll('input[name="models"]:checked'))
                       .map(checkbox => checkbox.value);
        },
        
        reloadModels: function() {
            loadModels();
        }
    };
})(); 