/**
 * RAG Search System - Collections Module
 * Handles collection management and AI queries
 */

import * as API from './api.js';
import * as UI from './ui.js';

/**
 * Initialize collections module
 */
export function init() {
  // AI query button
  document.getElementById('askBtn').addEventListener('click', handleAIQuery);
  
  // Enter key for AI query
  document.getElementById('askQuery').addEventListener('keypress', e => {
    if (e.key === 'Enter') {
      document.getElementById('askBtn').click();
    }
  });
  
  // Clear button
  document.getElementById('clearAskBtn').addEventListener('click', clearAIQuery);
  
  // Listen for collection changes
  window.addEventListener('collectionsChanged', refreshCollections);
  
  // Initial load of collections
  refreshCollections();
}

/**
 * Refresh collections in dropdowns
 */
export async function refreshCollections() {
  try {
    const data = await API.getCollections();
    
    // Update both search and ask dropdowns
    const askSelect = document.getElementById('askCollection');
    
    const options = data.collections.map(c => ({
      value: c.name,
      text: c.name + (c.loaded ? ' ✓' : '')
    }));
    
    UI.populateSelect(askSelect, options, '-- Välj samling --');
    
  } catch (error) {
    console.error('Error refreshing collections:', error);
    UI.showError('Kunde inte ladda samlingar', 'collections');
  }
}

/**
 * Handle AI query
 */
async function handleAIQuery() {
  const collection = document.getElementById('askCollection').value;
  const query = document.getElementById('askQuery').value.trim();
  const k = document.getElementById('askK').value;
  const llmBackend = document.getElementById('llmBackend').value;
  
  if (!collection) {
    UI.showError('Välj en samling', 'AI-fråga');
    return;
  }
  
  if (!query) {
    UI.showError('Skriv en fråga', 'AI-fråga');
    return;
  }
  
  const btn = document.getElementById('askBtn');
  const answerContainer = document.getElementById('askAnswer');
  const loadingContainer = document.getElementById('askLoading');
  const answerText = document.getElementById('askAnswerText');
  const sourcesContainer = document.getElementById('askSources');
  
  UI.hideLoading(answerContainer);
  UI.showLoading(loadingContainer);
  UI.setButtonEnabled(btn, false);
  
  try {
    const data = await API.askAI(collection, query, k, llmBackend);
    
    if (data.success === false) {
      UI.showError(data);
      answerText.textContent = 'AI-frågan misslyckades. Se felmeddelandet ovan.';
      sourcesContainer.innerHTML = '';
    } else {
      UI.displayAIAnswer(
        answerContainer,
        answerText,
        sourcesContainer,
        data.answer,
        data.sources
      );
    }
    
  } catch (error) {
    console.error('Error in AI query:', error);
    UI.showError(error.message, 'AI-fråga');
    answerText.textContent = `Fel: ${error.message}`;
    sourcesContainer.innerHTML = '';
    UI.showLoading(answerContainer);
  } finally {
    UI.hideLoading(loadingContainer);
    UI.showLoading(answerContainer);
    UI.setButtonEnabled(btn, true);
  }
}

/**
 * Clear AI query inputs and results
 */
function clearAIQuery() {
  UI.clearSelect(document.getElementById('askCollection'));
  UI.clearInput(document.getElementById('askQuery'));
  
  const answerContainer = document.getElementById('askAnswer');
  const loadingContainer = document.getElementById('askLoading');
  const answerText = document.getElementById('askAnswerText');
  const sourcesContainer = document.getElementById('askSources');
  
  UI.hideLoading(answerContainer);
  UI.hideLoading(loadingContainer);
  answerText.textContent = '';
  sourcesContainer.innerHTML = '';
}
