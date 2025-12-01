/**
 * RAG Search System - Main Entry Point
 * Initializes all modules and sets up the application
 */

import * as PDF from './pdf.js';
import * as URL from './url.js';
import * as Collections from './collections.js';

/**
 * Initialize the application
 */
function init() {
  console.log('RAG Search System - Initializing...');
  
  // Initialize all modules
  PDF.init();
  URL.init();
  Collections.init();
  
  console.log('RAG Search System - Ready!');
}

// Initialize when DOM is loaded
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
