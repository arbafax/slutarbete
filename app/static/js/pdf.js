/**
 * RAG Search System - PDF Module
 * Handles PDF upload and processing
 */

import * as API from './api.js';
import * as UI from './ui.js';

let currentMode = 'new';

/**
 * Initialize PDF module
 */
export function init() {
  // Mode switching
  document.getElementById('pdfModeNew').addEventListener('click', () => switchMode('new'));
  document.getElementById('pdfModeAdd').addEventListener('click', () => switchMode('add'));
  
  // Drag and drop for new collection
  const pdfDropzone = document.getElementById('pdfDropzone');
  const pdfFileInput = document.getElementById('pdfFile');
  
  setupDragAndDrop(pdfDropzone);
  pdfDropzone.addEventListener('drop', e => handleFiles(e.dataTransfer.files));
  pdfFileInput.addEventListener('change', e => handleFiles(e.target.files));
  
  // Drag and drop for existing collection
  const addPdfDropzone = document.getElementById('addPdfDropzone');
  const addPdfFileInput = document.getElementById('addPdfFile');
  
  setupDragAndDrop(addPdfDropzone);
  addPdfDropzone.addEventListener('drop', e => handleAddFiles(e.dataTransfer.files));
  addPdfFileInput.addEventListener('change', e => handleAddFiles(e.target.files));
  
  // Clear button
  document.getElementById('clearPdfBtn').addEventListener('click', clearAll);
}

/**
 * Setup drag and drop events
 */
function setupDragAndDrop(dropzone) {
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, e => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });
  
  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, e => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });
}

/**
 * Switch between new/add modes
 */
function switchMode(mode) {
  currentMode = mode;
  
  const newBtn = document.getElementById('pdfModeNew');
  const addBtn = document.getElementById('pdfModeAdd');
  const newMode = document.getElementById('pdfNewMode');
  const addMode = document.getElementById('pdfAddMode');
  
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
 * Handle PDF files for new collection
 */
async function handleFiles(files) {
  if (!files || files.length === 0) {
    UI.showError('Välj minst en PDF-fil', 'PDF-val');
    return;
  }
  
  const collectionName = document.getElementById('pdfCollectionName').value.trim();
  await processFiles(files, collectionName);
}

/**
 * Handle PDF files for existing collection
 */
async function handleAddFiles(files) {
  const collection = document.getElementById('existingPdfCollectionSelect').value;
  
  if (!collection) {
    UI.showError('Välj en samling', 'PDF-tillägg');
    return;
  }
  
  if (!files || files.length === 0) {
    UI.showError('Välj minst en PDF-fil', 'PDF-val');
    return;
  }
  
  await processFiles(files, collection);
}

/**
 * Process PDF files
 */
async function processFiles(files, collectionName) {
  const backend = document.getElementById(
    currentMode === 'new' ? 'pdfEmbedBackend' : 'addPdfEmbedBackend'
  ).value;
  const chunkSize = document.getElementById(
    currentMode === 'new' ? 'pdfChunkSize' : 'addPdfChunkSize'
  ).value;
  
  const progress = document.getElementById('pdfProgress');
  const progressBar = document.getElementById('pdfProgressBar');
  const progressText = document.getElementById('pdfProgressText');
  
  UI.showProgress(progress);
  
  let completed = 0;
  let lastData = null;
  const errors = [];
  
  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      errors.push(`${file.name}: Inte en PDF-fil`);
      continue;
    }
    
    console.info('r143');

      try {
      UI.updateProgress(
        progressBar, 
        progressText, 
        completed, 
        files.length, 
        `${completed} / ${files.length} - Laddar upp ${file.name}`
      );
      
      const data = await API.uploadPdf(file, collectionName, backend, chunkSize);
      
      if (data.success === false) {
        UI.showError(data);
        errors.push(`${file.name}: Processering misslyckades`);
        continue;
      }
      
      lastData = data;
      completed++;
      UI.updateProgress(progressBar, progressText, completed, files.length);
      
    } catch (error) {
      console.error(`Fel vid ${file.name}:`, error);
      errors.push(`${file.name}: ${error.message}`);
    }
  }
  
  if (errors.length > 0) {
    UI.showError(`Några filer kunde inte processas:\n\n${errors.join('\n')}`);
  }
  
  progressText.textContent = `Klart! ${completed} / ${files.length} PDFs indexerade`;
  
  if (lastData) {
    displayResults(lastData);
  }
  
  // Clear inputs
  if (currentMode === 'new') {
    UI.clearInput(document.getElementById('pdfCollectionName'));
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
  document.getElementById('pdfCollectionNameDisplay').textContent = data.collection_name || '-';
  document.getElementById('pdfRecordCount').textContent = data.record_count || data.total_records || 0;
  document.getElementById('pdfVectorCount').textContent = data.vector_store?.stats?.total_vectors || data.total_vectors || 0;
  document.getElementById('pdfResult').style.display = 'block';
  
  if (data.indexed_pdfs && data.indexed_pdfs.length > 0) {
    const listContainer = document.getElementById('pdfListContainer');
    const listDisplay = document.getElementById('pdfListDisplay');
    
    UI.toggleVisibility(listContainer, true);
    
    listDisplay.innerHTML = data.indexed_pdfs.map(pdf => 
      `<div class="url-item">
        <span class="material-icons" style="font-size: 16px; color: var(--secondary);">description</span>
        <span class="url-item-text">${pdf}</span>
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
    const select = document.getElementById('existingPdfCollectionSelect');
    
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
 * Clear all PDF inputs and results
 */
function clearAll() {
  UI.clearInput(document.getElementById('pdfCollectionName'));
  UI.clearInput(document.getElementById('pdfFile'));
  UI.clearInput(document.getElementById('addPdfFile'));
  UI.clearSelect(document.getElementById('existingPdfCollectionSelect'));
  
  const progress = document.getElementById('pdfProgress');
  const progressBar = document.getElementById('pdfProgressBar');
  const progressText = document.getElementById('pdfProgressText');
  
  UI.hideProgress(progress);
  UI.resetProgress(progressBar, progressText);
  
  document.getElementById('pdfResult').style.display = 'none';
  document.getElementById('pdfListContainer').classList.add('hidden');
  document.getElementById('pdfListDisplay').innerHTML = '';
}

/**
 * Get current mode
 */
export function getCurrentMode() {
  return currentMode;
}
