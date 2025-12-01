/**
 * RAG Search System - API Module
 * Handles all API communication with the backend
 */

/**
 * Upload PDF file to server
 * @param {File} file - PDF file to upload
 * @param {string} collectionName - Name of collection (optional)
 * @param {string} backend - Embedding backend to use
 * @param {number} chunkSize - Chunk size for text splitting
 * @returns {Promise<Object>} Server response
 */
export async function uploadPdf(file, collectionName, backend, chunkSize) {
  const form = new FormData();
  form.append('file', file);
  
  let url = `/api/upload_pdf?embed_backend=${backend}&max_tokens_per_chunk=${chunkSize}`;
  if (collectionName) {
    url += `&collection_name=${encodeURIComponent(collectionName)}`;
  }
  
  const response = await fetch(url, { 
    method: 'POST', 
    body: form 
  });
  
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  
  return await response.json();
}

/**
 * Fetch URL and add to collection
 * @param {string} url - URL to fetch
 * @param {string} collectionName - Name of collection (optional)
 * @param {string} backend - Embedding backend to use
 * @returns {Promise<Object>} Server response
 */
export async function fetchUrl(url, collectionName, backend) {
  const payload = { 
    url: url.trim(), 
    embed_backend: backend 
  };
  
  if (collectionName) {
    payload.collection_name = collectionName;
  }

  const response = await fetch('/api/fetch_url', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  
  return await response.json();
}

/**
 * Search in a collection
 * @param {string} collection - Collection name
 * @param {string} query - Search query
 * @param {number} k - Number of results
 * @param {string} embedBackend - Embedding backend (optional)
 * @returns {Promise<Object>} Search results
 */
export async function search(collection, query, k, embedBackend = null) {
  const payload = { 
    collection, 
    query, 
    k: parseInt(k) 
  };
  
  if (embedBackend) {
    payload.embed_backend = embedBackend;
  }

  const response = await fetch('/api/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  
  return await response.json();
}

/**
 * Ask AI a question about a collection
 * @param {string} collection - Collection name
 * @param {string} query - Question to ask
 * @param {number} k - Number of sources to use
 * @param {string} llmBackend - LLM backend to use
 * @param {string} embedBackend - Embedding backend (optional)
 * @returns {Promise<Object>} AI response with sources
 */
export async function askAI(collection, query, k, llmBackend, embedBackend = null) {
  const payload = { 
    collection, 
    query, 
    k: parseInt(k),
    llm_backend: llmBackend
  };
  
  if (embedBackend) {
    payload.embed_backend = embedBackend;
  }

  const response = await fetch('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  
  return await response.json();
}

/**
 * Get list of all collections
 * @returns {Promise<Object>} List of collections with stats
 */
export async function getCollections() {
  const response = await fetch('/api/collections');
  
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  
  return await response.json();
}

/**
 * Delete a collection
 * @param {string} collectionName - Name of collection to delete
 * @returns {Promise<Object>} Deletion result
 */
export async function deleteCollection(collectionName) {
  const response = await fetch(`/api/collection/${encodeURIComponent(collectionName)}`, {
    method: 'DELETE'
  });
  
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  
  return await response.json();
}

/**
 * Health check endpoint
 * @returns {Promise<Object>} Health status
 */
export async function healthCheck() {
  const response = await fetch('/api/health');
  
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  
  return await response.json();
}
