/**
 * RAG Search System - Main Entry Point
 * Initializes all modules and sets up the application
 */

import * as Tabs from './tabs.js';
import * as PDF from './pdf.js';
import * as URL from './url.js';
import * as Collections from './collections.js';
import * as Search from './search.js';

/**
 * Initialize the application
 */
function init() {
  console.log('RAG Search System - Initializing...');
  
  // Initialize all modules
  Tabs.init();
  PDF.init();
  URL.init();
  Collections.init();
  Search.init();
  
  console.log('RAG Search System - Ready!');
}

// Initialize when DOM is loaded
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
