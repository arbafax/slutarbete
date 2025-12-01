/**
 * RAG Search System - UI Module
 * Handles UI updates, error display, and progress indicators
 */

/**
 * Show error message to user
 * @param {Object|string} errorData - Error data from server or error message
 * @param {string} context - Context where error occurred (optional)
 */
export function showError(errorData, context = '') {
  let message = '';
  
  if (typeof errorData === 'object' && errorData.error) {
    // Structured error from server
    message = errorData.error.message || errorData.error.details || 'Ett okänt fel uppstod';
    
    if (errorData.error.context) {
      message = `FEL VID ${errorData.error.context.toUpperCase()}:\n\n${message}`;
    }
  } else if (typeof errorData === 'object' && errorData.detail) {
    // HTTPException from FastAPI
    message = errorData.detail;
  } else if (typeof errorData === 'string') {
    // Simple string
    message = errorData;
  } else {
    message = 'Ett okänt fel uppstod';
  }
  
  // Show alert with error
  alert(message);
  
  // Also log to console for debugging
  console.error('Error:', errorData);
}

/**
 * Handle fetch errors and extract error data
 * @param {Response} response - Fetch response object
 * @returns {Promise<Object>} Error data
 */
export async function handleFetchError(response) {
  let errorData;
  
  try {
    errorData = await response.json();
  } catch (e) {
    errorData = { 
      error: { 
        message: `HTTP ${response.status}: ${response.statusText}` 
      } 
    };
  }
  
  // Show error to user
  if (errorData.success === false || errorData.error) {
    showError(errorData);
  } else {
    showError(errorData);
  }
  
  return errorData;
}

/**
 * Update progress bar
 * @param {HTMLElement} progressBar - Progress bar element
 * @param {HTMLElement} progressText - Progress text element
 * @param {number} current - Current progress
 * @param {number} total - Total items
 * @param {string} message - Optional message to display
 */
export function updateProgress(progressBar, progressText, current, total, message = '') {
  const percentage = total > 0 ? (current / total) * 100 : 0;
  progressBar.style.width = `${percentage}%`;
  
  if (message) {
    progressText.textContent = message;
  } else {
    progressText.textContent = `${current} / ${total}`;
  }
}

/**
 * Show progress indicator
 * @param {HTMLElement} progressContainer - Progress container element
 */
export function showProgress(progressContainer) {
  progressContainer.style.display = 'block';
}

/**
 * Hide progress indicator
 * @param {HTMLElement} progressContainer - Progress container element
 * @param {number} delay - Delay before hiding (ms)
 */
export function hideProgress(progressContainer, delay = 0) {
  if (delay > 0) {
    setTimeout(() => {
      progressContainer.style.display = 'none';
    }, delay);
  } else {
    progressContainer.style.display = 'none';
  }
}

/**
 * Reset progress indicator
 * @param {HTMLElement} progressBar - Progress bar element
 * @param {HTMLElement} progressText - Progress text element
 */
export function resetProgress(progressBar, progressText) {
  progressBar.style.width = '0%';
  progressText.textContent = '0 / 0';
}

/**
 * Display search results
 * @param {HTMLElement} container - Container element for results
 * @param {Array} results - Array of search results
 * @param {number} totalCount - Total number of results
 */
export function displaySearchResults(container, results, totalCount) {
  if (!results || results.length === 0) {
    container.innerHTML = '<div class="empty-state"><span class="material-icons">search_off</span><p>Inga resultat.</p></div>';
    return;
  }
  
  let html = `<h3>${totalCount} resultat</h3>`;
  results.forEach(r => {
    const heading = r.heading || 'Utan titel';
    const score = typeof r.score === 'number' ? r.score.toFixed(4) : r.score;
    html += `
      <div class="search-result">
        <div class="search-result-header">
          <h4 class="search-result-title">${escapeHtml(heading)}</h4>
          <span class="search-result-score">${score}</span>
        </div>
        <div class="search-result-text">${escapeHtml(r.markdown || r.text || '')}</div>
      </div>
    `;
  });
  container.innerHTML = html;
}

/**
 * Display AI answer with sources
 * @param {HTMLElement} answerContainer - Answer container element
 * @param {HTMLElement} answerText - Answer text element
 * @param {HTMLElement} sourcesContainer - Sources container element
 * @param {string} answer - AI answer
 * @param {Array} sources - Array of sources
 */
export function displayAIAnswer(answerContainer, answerText, sourcesContainer, answer, sources) {
  answerText.textContent = answer;
  
  let srcHtml = '';
  if (sources && sources.length > 0) {
    sources.forEach(s => {
      const heading = s.heading || `Källa ${s.rank}`;
      const score = typeof s.score === 'number' ? s.score.toFixed(4) : s.score;
      srcHtml += `
        <div class="search-result">
          <div class="search-result-header">
            <h4 class="search-result-title">${escapeHtml(heading)}</h4>
            <span class="search-result-score">${score}</span>
          </div>
          <div class="search-result-text">${escapeHtml(s.preview || '')}</div>
        </div>
      `;
    });
  } else {
    srcHtml = '<p class="text-muted">Inga källor.</p>';
  }
  
  sourcesContainer.innerHTML = srcHtml;
  answerContainer.style.display = 'block';
}

/**
 * Show loading indicator
 * @param {HTMLElement} element - Element to show
 */
export function showLoading(element) {
  element.style.display = 'block';
}

/**
 * Hide loading indicator
 * @param {HTMLElement} element - Element to hide
 */
export function hideLoading(element) {
  element.style.display = 'none';
}

/**
 * Enable/disable button
 * @param {HTMLElement} button - Button element
 * @param {boolean} enabled - Whether button should be enabled
 */
export function setButtonEnabled(button, enabled) {
  button.disabled = !enabled;
}

/**
 * Clear input field
 * @param {HTMLElement} input - Input element to clear
 */
export function clearInput(input) {
  input.value = '';
}

/**
 * Clear select dropdown
 * @param {HTMLElement} select - Select element to clear
 */
export function clearSelect(select) {
  select.value = '';
}

/**
 * Populate select dropdown with options
 * @param {HTMLElement} select - Select element
 * @param {Array} options - Array of {value, text} objects
 * @param {string} placeholder - Placeholder text for first option
 * @param {string} currentValue - Currently selected value (optional)
 */
export function populateSelect(select, options, placeholder = '-- Välj --', currentValue = null) {
  const currentVal = currentValue || select.value;
  select.innerHTML = `<option value="">${placeholder}</option>`;
  
  options.forEach(opt => {
    const option = document.createElement('option');
    option.value = opt.value;
    option.textContent = opt.text;
    select.appendChild(option);
  });
  
  if (currentVal) {
    select.value = currentVal;
  }
}

/**
 * Escape HTML to prevent XSS
 * @param {string} text - Text to escape
 * @returns {string} Escaped text
 */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/**
 * Toggle visibility of element
 * @param {HTMLElement} element - Element to toggle
 * @param {boolean} visible - Whether element should be visible
 */
export function toggleVisibility(element, visible) {
  if (visible) {
    element.classList.remove('hidden');
  } else {
    element.classList.add('hidden');
  }
}

/**
 * Switch active mode button
 * @param {HTMLElement} activeButton - Button to activate
 * @param {Array<HTMLElement>} allButtons - All mode buttons
 */
export function switchModeButton(activeButton, allButtons) {
  allButtons.forEach(btn => btn.classList.remove('active'));
  activeButton.classList.add('active');
}
