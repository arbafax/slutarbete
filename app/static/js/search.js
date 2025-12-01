/**
 * RAG Search System - Search Module
 * Handles semantic search functionality
 */

import * as API from './api.js';
import * as UI from './ui.js';

/**
 * Initialize search module
 */
export function init() {
  // Search button
  document.getElementById('searchBtn').addEventListener('click', handleSearch);
  
  // Refresh collections button
  document.getElementById('refreshCollections').addEventListener('click', refreshCollections);
  
  // Enter key for search
  document.getElementById('searchQuery').addEventListener('keypress', e => {
    if (e.key === 'Enter') {
      document.getElementById('searchBtn').click();
    }
  });
  
  // Clear button
  document.getElementById('clearSearchBtn').addEventListener('click', clearSearch);
  
  // Listen for collection changes
  window.addEventListener('collectionsChanged', refreshCollections);
  
  // Initial load of collections
  refreshCollections();
}

/**
 * Handle search
 */
async function handleSearch() {
  const collection = document.getElementById('searchCollection').value;
  const query = document.getElementById('searchQuery').value.trim();
  const k = document.getElementById('searchK').value;
  const resultsContainer = document.getElementById('searchResults');
  
  if (!collection) {
    UI.showError('Välj en samling', 'sökning');
    return;
  }
  
  if (!query) {
    UI.showError('Ange sökfråga', 'sökning');
    return;
  }
  
  resultsContainer.innerHTML = '<p><span class="spinner"></span> Söker...</p>';
  
  try {
    const data = await API.search(collection, query, k);
    
    if (data.success === false) {
      UI.showError(data);
      resultsContainer.innerHTML = '<p style="color:var(--error);">Sökningen misslyckades</p>';
      return;
    }
    
    UI.displaySearchResults(resultsContainer, data.results, data.results_count);
    
  } catch (error) {
    console.error('Error in search:', error);
    UI.showError(error.message, 'sökning');
    resultsContainer.innerHTML = `<p style="color:var(--error);">Fel: ${error.message}</p>`;
  }
}

/**
 * Refresh collections dropdown
 */
async function refreshCollections() {
  try {
    const data = await API.getCollections();
    const select = document.getElementById('searchCollection');
    
    const options = data.collections.map(c => ({
      value: c.name,
      text: c.name + (c.loaded ? ' ✓' : '')
    }));
    
    UI.populateSelect(select, options, '-- Välj samling --');
    
  } catch (error) {
    console.error('Error refreshing collections:', error);
  }
}

/**
 * Clear search inputs and results
 */
function clearSearch() {
  UI.clearSelect(document.getElementById('searchCollection'));
  UI.clearInput(document.getElementById('searchQuery'));
  document.getElementById('searchResults').innerHTML = '';
}
