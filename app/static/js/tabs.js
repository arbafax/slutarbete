/**
 * RAG Search System - Tabs Module
 * Handles tab switching and navigation
 */

/**
 * Initialize tabs module
 */
export function init() {
  const tabButtons = document.querySelectorAll('.tab-button');
  
  tabButtons.forEach(button => {
    button.addEventListener('click', () => {
      const targetTab = button.getAttribute('data-tab');
      switchTab(targetTab);
    });
  });
}

/**
 * Switch to a specific tab
 * @param {string} tabName - Name of tab to switch to (pdf, url, ask, search)
 */
export function switchTab(tabName) {
  // Deactivate all tabs
  document.querySelectorAll('.tab-button').forEach(btn => {
    btn.classList.remove('active');
  });
  
  document.querySelectorAll('.tab-panel').forEach(panel => {
    panel.classList.remove('active');
  });
  
  // Activate selected tab
  const targetButton = document.querySelector(`[data-tab="${tabName}"]`);
  const targetPanel = document.getElementById(`tab-${tabName}`);
  
  if (targetButton && targetPanel) {
    targetButton.classList.add('active');
    targetPanel.classList.add('active');
  }
}

/**
 * Get currently active tab
 * @returns {string} Name of active tab
 */
export function getActiveTab() {
  const activeButton = document.querySelector('.tab-button.active');
  return activeButton ? activeButton.getAttribute('data-tab') : null;
}
