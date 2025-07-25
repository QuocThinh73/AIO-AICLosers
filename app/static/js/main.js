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

// Initialize objects section (DISABLED - feature removed)
window.AppData.selectedObjects = [];
window.AppData.availableObjects = [];

// Add notification about removed object filtering feature
if (document.getElementById('selectedObjectsTags')) {
    document.getElementById('selectedObjectsTags').innerHTML = 
        '<div class="notification" style="padding: 5px 10px; background-color: #f8d7da; color: #721c24; border-radius: 4px; font-size: 0.8rem;">' +
        '<i class="fas fa-info-circle"></i> Object filtering has been removed from this version' +
        '</div>';
}

// Disable object input
if (document.getElementById('objectInput')) {
    document.getElementById('objectInput').disabled = true;
    document.getElementById('objectInput').placeholder = "Object filtering disabled";
}

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
