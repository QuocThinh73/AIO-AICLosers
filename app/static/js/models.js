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
            const label = document.createElement('label');
            
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.value = model;
            checkbox.name = 'models';
            checkbox.checked = true;

            const span = document.createElement('span');
            span.textContent = model;

            label.appendChild(checkbox);
            label.appendChild(span);
            modelSelect.appendChild(label);
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