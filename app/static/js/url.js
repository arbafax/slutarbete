/**
 * RAG Search System - URL Module
 * Handles URL fetching and processing
 */

import * as API from './api.js';
import * as UI from './ui.js';

let currentMode = 'new';

/**
 * Initialize URL module
 */
export function init() {
  // Mode switching
  document.getElementById('urlModeNew').addEventListener('click', () => switchMode('new'));
  document.getElementById('urlModeAdd').addEventListener('click', () => switchMode('add'));
  
  // Fetch buttons
  document.getElementById('urlFetchBtn').addEventListener('click', handleNewFetch);
  document.getElementById('addUrlFetchBtn').addEventListener('click', handleAddFetch);
  
  // Clear button
  document.getElementById('clearUrlBtn').addEventListener('click', clearAll);
}

/**
 * Switch between new/add modes
 */
function switchMode(mode) {
  currentMode = mode;
  
  const newBtn = document.getElementById('urlModeNew');
  const addBtn = document.getElementById('urlModeAdd');
  const newMode = document.getElementById('urlNewMode');
  const addMode = document.getElementById('urlAddMode');
  
  UI.switchModeButton(
    mode === 'new' ? newBtn : addBtn,
    [newBtn, addBtn]
  );
  
  UI.toggleVisibility(newMode, mode === 'new');
  UI.toggleVisibility(addMode, mode === 'add');
  
  if (mode === 'add') {
    refreshExistingCollections();
  }
}

/**
 * Handle fetch for new collection
 */
async function handleNewFetch() {
  const urls = document.getElementById('urlInput').value
    .trim()
    .split('\n')
    .filter(u => u.trim());
  
  const collectionName = document.getElementById('urlCollectionName').value.trim();
  const backend = document.getElementById('urlEmbedBackend').value;
  
  if (urls.length === 0) {
    UI.showError('Ange minst en URL', 'URL-val');
    return;
  }
  
  await fetchUrls(urls, collectionName, backend);
}

/**
 * Handle fetch for existing collection
 */
async function handleAddFetch() {
  const urls = document.getElementById('addUrlInput').value
    .trim()
    .split('\n')
    .filter(u => u.trim());
  
  const collection = document.getElementById('existingUrlCollectionSelect').value;
  const backend = document.getElementById('addUrlEmbedBackend').value;
  
  if (!collection) {
    UI.showError('Välj en samling', 'URL-tillägg');
    return;
  }
  
  if (urls.length === 0) {
    UI.showError('Ange minst en URL', 'URL-val');
    return;
  }
  
  await fetchUrls(urls, collection, backend);
}

/**
 * Fetch and process URLs
 */
async function fetchUrls(urls, collectionName, backend) {
  const btn = document.getElementById(
    currentMode === 'new' ? 'urlFetchBtn' : 'addUrlFetchBtn'
  );
  const progress = document.getElementById('urlProgress');
  const progressBar = document.getElementById('urlProgressBar');
  const progressText = document.getElementById('urlProgressText');
  
  UI.setButtonEnabled(btn, false);
  UI.showProgress(progress);
  
  let completed = 0;
  let lastData = null;
  const errors = [];
  
  for (const url of urls) {
    if (!url.trim()) continue;
    
    try {
      UI.updateProgress(
        progressBar,
        progressText,
        completed,
        urls.length,
        `${completed} / ${urls.length} - Hämtar ${url}`
      );
      
      const data = await API.fetchUrl(url, collectionName, backend);
      
      if (data.success === false) {
        UI.showError(data);
        errors.push(`${url}: Processering misslyckades`);
        continue;
      }
      
      lastData = data;
      completed++;
      UI.updateProgress(progressBar, progressText, completed, urls.length);
      
    } catch (error) {
      console.error(`Fel vid ${url}:`, error);
      errors.push(`${url}: ${error.message}`);
    }
  }
  
  if (errors.length > 0) {
    UI.showError(`Några URLs kunde inte processas:\n\n${errors.join('\n')}`);
  }
  
  progressText.textContent = `Klart! ${completed} / ${urls.length} URLs indexerade`;
  UI.setButtonEnabled(btn, true);
  
  if (lastData) {
    displayResults(lastData);
  }
  
  // Clear inputs
  if (currentMode === 'new') {
    UI.clearInput(document.getElementById('urlInput'));
    UI.clearInput(document.getElementById('urlCollectionName'));
  } else {
    UI.clearInput(document.getElementById('addUrlInput'));
  }
  
  // Notify collections module to refresh
  window.dispatchEvent(new Event('collectionsChanged'));
  
  // Hide progress after delay
  setTimeout(() => {
    UI.hideProgress(progress);
    UI.resetProgress(progressBar, progressText);
  }, 3000);
}

/**
 * Display processing results
 */
function displayResults(data) {
  document.getElementById('urlCollectionNameDisplay').textContent = data.collection_name || '-';
  document.getElementById('urlRecordCount').textContent = data.record_count || data.total_records || 0;
  document.getElementById('urlVectorCount').textContent = data.vector_store?.stats?.total_vectors || data.total_vectors || 0;
  document.getElementById('urlResult').style.display = 'block';
  
  if (data.indexed_urls && data.indexed_urls.length > 0) {
    const listContainer = document.getElementById('urlListContainer');
    const listDisplay = document.getElementById('urlListDisplay');
    
    UI.toggleVisibility(listContainer, true);
    
    listDisplay.innerHTML = data.indexed_urls.map(url => 
      `<div class="url-item">
        <span class="material-icons" style="font-size: 16px; color: var(--secondary);">link</span>
        <span class="url-item-text">${url}</span>
      </div>`
    ).join('');
  }
}

/**
 * Refresh existing collections dropdown
 */
async function refreshExistingCollections() {
  try {
    const data = await API.getCollections();
    const select = document.getElementById('existingUrlCollectionSelect');
    
    const options = data.collections.map(c => ({
      value: c.name,
      text: c.name
    }));
    
    UI.populateSelect(select, options, '-- Välj samling --');
    
  } catch (error) {
    console.error('Error refreshing collections:', error);
  }
}

/**
 * Clear all URL inputs and results
 */
function clearAll() {
  UI.clearInput(document.getElementById('urlInput'));
  UI.clearInput(document.getElementById('urlCollectionName'));
  UI.clearInput(document.getElementById('addUrlInput'));
  UI.clearSelect(document.getElementById('existingUrlCollectionSelect'));
  
  const progress = document.getElementById('urlProgress');
  const progressBar = document.getElementById('urlProgressBar');
  const progressText = document.getElementById('urlProgressText');
  
  UI.hideProgress(progress);
  UI.resetProgress(progressBar, progressText);
  
  document.getElementById('urlResult').style.display = 'none';
  document.getElementById('urlListContainer').classList.add('hidden');
  document.getElementById('urlListDisplay').innerHTML = '';
}

/**
 * Get current mode
 */
export function getCurrentMode() {
  return currentMode;
}
