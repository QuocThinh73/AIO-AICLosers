// ===================================================================
// MAIN.JS - Entry point and application initialization
// ===================================================================

// Import modules (loaded via script tags)
// - search.js: Handle search and display results
// - models.js: Handle model selection
// - objects.js: Handle object selection  
// - modal.js: Handle modal display of image details

// Global variables for the application
window.AppData = {
    availableModels: [],
    availableObjects: [],
    selectedObjects: []
};

// Main DOM elements
window.AppElements = {
    searchForm: document.getElementById('searchForm'),
    searchInput: document.getElementById('searchInput'),
    ocrInput: document.getElementById('ocrInput'),
    topKInput: document.getElementById('topK'),
    resultsDiv: document.getElementById('results'),
    modelSelect: document.getElementById('modelSelect'),
    objectInput: document.getElementById('objectInput'),
    objectDropdown: document.getElementById('objectDropdown'),
    selectedObjectsTags: document.getElementById('selectedObjectsTags')
};

// Application initialization function
function initApp() {
    // Check required elements
    if (!window.AppElements.searchForm || !window.AppElements.searchInput || 
        !window.AppElements.resultsDiv || !window.AppElements.modelSelect) {
        return;
    }
    
    // Initialize modules
    if (window.SearchModule) {
        window.SearchModule.init();
    }
    
    if (window.ModelsModule) {
        window.ModelsModule.init();
    }
    
    if (window.ObjectsModule) {
        window.ObjectsModule.init();
    }
    
    if (window.ModalModule) {
        window.ModalModule.init();
    }
}

// Start application when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}
