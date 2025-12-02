/**
 * RAG Search System - Collections Management Module
 * Handles the Collections tab - list and detailed view
 */

import * as API from './api.js';
import * as UI from './ui.js';

let currentSelectedCollection = null;

/**
 * Initialize collections management module
 */
export function init() {
  // Refresh button
  document.getElementById('refreshCollectionsBtn').addEventListener('click', loadCollectionsList);
  
  // Delete button
  document.getElementById('deleteCollectionBtn').addEventListener('click', handleDeleteCollection);
  
  // Listen for collection changes from other tabs
  window.addEventListener('collectionsChanged', loadCollectionsList);
  
  // Initial load
  loadCollectionsList();
}

/**
 * Load and display collections list
 */
async function loadCollectionsList() {
  const listContainer = document.getElementById('collectionsList');
  
  try {
    listContainer.innerHTML = '<p><span class="spinner"></span> Laddar samlingar...</p>';
    
    const data = await API.getCollections();
    
    if (!data.collections || data.collections.length === 0) {
      listContainer.innerHTML = `
        <div class="empty-state">
          <span class="material-icons">folder_off</span>
          <p>Inga samlingar hittades</p>
          <p class="text-muted">Skapa en ny samling genom att ladda upp en PDF eller extrahera från en URL.</p>
        </div>
      `;
      return;
    }
    
    // Sort collections by name
    const collections = data.collections.sort((a, b) => a.name.localeCompare(b.name));
    
    // Render collection cards
    let html = '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px;">';
    
    collections.forEach(collection => {
      const isActive = currentSelectedCollection === collection.name;
      const activeClass = isActive ? 'collection-card-active' : '';
      
      html += `
        <div class="collection-card ${activeClass}" data-collection="${collection.name}">
          <div style="display: flex; justify-content: space-between; align-items: start;">
            <div style="flex: 1;">
              <h4 style="margin: 0 0 8px; display: flex; align-items: center; gap: 8px;">
                <span class="material-icons" style="font-size: 20px;">folder</span>
                ${escapeHtml(collection.name)}
              </h4>
              <div style="display: flex; gap: 16px; font-size: 13px; color: var(--on-surface-medium);">
                <span title="PDFs"><span class="material-icons" style="font-size: 14px; vertical-align: middle;">picture_as_pdf</span> ${collection.pdf_count || 0}</span>
                <span title="URLs"><span class="material-icons" style="font-size: 14px; vertical-align: middle;">link</span> ${collection.url_count || 0}</span>
                <span title="Chunks"><span class="material-icons" style="font-size: 14px; vertical-align: middle;">category</span> ${collection.stats?.total_records || 0}</span>
              </div>
            </div>
            ${collection.loaded ? '<span class="material-icons" style="color: var(--secondary);" title="Laddad i minnet">check_circle</span>' : ''}
          </div>
        </div>
      `;
    });
    
    html += '</div>';
    listContainer.innerHTML = html;
    
    // Add click handlers
    document.querySelectorAll('.collection-card').forEach(card => {
      card.addEventListener('click', () => {
        const collectionName = card.getAttribute('data-collection');
        selectCollection(collectionName);
      });
    });
    
  } catch (error) {
    console.error('Error loading collections:', error);
    UI.showError(error.message, 'samlings-listning');
    listContainer.innerHTML = `<p style="color:var(--error);">Kunde inte ladda samlingar: ${error.message}</p>`;
  }
}

/**
 * Select a collection and load its details
 */
async function selectCollection(collectionName) {
  currentSelectedCollection = collectionName;
  
  // Update active state in list
  document.querySelectorAll('.collection-card').forEach(card => {
    if (card.getAttribute('data-collection') === collectionName) {
      card.classList.add('collection-card-active');
    } else {
      card.classList.remove('collection-card-active');
    }
  });
  
  // Load detailed metadata
  await loadCollectionDetails(collectionName);
}

/**
 * Load and display detailed metadata for a collection
 */
async function loadCollectionDetails(collectionName) {
  const detailsContainer = document.getElementById('collectionDetails');
  
  try {
    detailsContainer.style.display = 'block';
    
    // Show loading state
    document.getElementById('detailCollectionName').textContent = 'Laddar...';
    
    // Fetch detailed metadata
    const response = await fetch(`/api/collection/${encodeURIComponent(collectionName)}/metadata`);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    
    if (data.success === false) {
      UI.showError(data);
      return;
    }
    
    const metadata = data.metadata;
    
    // Update stats
    document.getElementById('detailCollectionName').textContent = metadata.name;
    document.getElementById('detailChunkCount').textContent = metadata.total_records || 0;
    document.getElementById('detailVectorCount').textContent = metadata.stats?.total_vectors || 0;
    document.getElementById('detailDimensions').textContent = metadata.stats?.dimension || '-';
    
    // Update counts
    document.getElementById('detailPdfCount').textContent = metadata.pdf_count || 0;
    document.getElementById('detailUrlCount').textContent = metadata.url_count || 0;
    
    // Update PDF list
    const pdfList = document.getElementById('detailPdfList');
    if (metadata.indexed_pdfs && metadata.indexed_pdfs.length > 0) {
      pdfList.innerHTML = metadata.indexed_pdfs.map(pdf => 
        `<div class="url-item">
          <span class="material-icons" style="font-size: 16px; color: var(--error);">picture_as_pdf</span>
          <span class="url-item-text">${escapeHtml(pdf)}</span>
        </div>`
      ).join('');
    } else {
      pdfList.innerHTML = '<p class="text-muted" style="margin: 8px;">Inga PDFs</p>';
    }
    
    // Update URL list
    const urlList = document.getElementById('detailUrlList');
    if (metadata.indexed_urls && metadata.indexed_urls.length > 0) {
      urlList.innerHTML = metadata.indexed_urls.map(url => 
        `<div class="url-item">
          <span class="material-icons" style="font-size: 16px; color: var(--primary);">link</span>
          <span class="url-item-text" title="${escapeHtml(url)}">${escapeHtml(url)}</span>
        </div>`
      ).join('');
    } else {
      urlList.innerHTML = '<p class="text-muted" style="margin: 8px;">Inga URLs</p>';
    }
    
    // Update timestamps
    document.getElementById('detailCreated').textContent = formatTimestamp(metadata.created);
    document.getElementById('detailModified').textContent = formatTimestamp(metadata.modified);
    
  } catch (error) {
    console.error('Error loading collection details:', error);
    UI.showError(error.message, 'metadata-hämtning');
    detailsContainer.style.display = 'none';
  }
}

/**
 * Handle delete collection
 */
async function handleDeleteCollection() {
  if (!currentSelectedCollection) {
    UI.showError('Ingen samling vald', 'borttagning');
    return;
  }
  
  const confirmed = confirm(
    `Är du säker på att du vill ta bort samlingen "${currentSelectedCollection}"?\n\n` +
    `Detta kommer att radera alla filer och kan inte ångras.`
  );
  
  if (!confirmed) {
    return;
  }
  
  try {
    await API.deleteCollection(currentSelectedCollection);
    
    // Hide details
    document.getElementById('collectionDetails').style.display = 'none';
    currentSelectedCollection = null;
    
    // Reload list
    await loadCollectionsList();
    
    // Notify other tabs
    window.dispatchEvent(new Event('collectionsChanged'));
    
    alert(`Samlingen har tagits bort.`);
    
  } catch (error) {
    console.error('Error deleting collection:', error);
    UI.showError(error.message, 'borttagning');
  }
}

/**
 * Format timestamp for display
 */
function formatTimestamp(isoString) {
  if (!isoString) return '-';
  
  try {
    const date = new Date(isoString);
    return date.toLocaleString('sv-SE', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    });
  } catch (e) {
    return isoString;
  }
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/**
 * Get currently selected collection
 */
export function getSelectedCollection() {
  return currentSelectedCollection;
}
