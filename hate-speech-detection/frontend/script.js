// ==================== CONFIGURATION ====================
const API_URL = 'http://localhost:8000';

// ==================== GLOBAL STATE ====================
let analyticsData = null;
let allUsers = [];
let currentChart = null;

// ==================== DOM ELEMENTS ====================
const elements = {
    // Tabs
    tabButtons: document.querySelectorAll('.tab-button'),
    tabContents: document.querySelectorAll('.tab-content'),
    
    // CSV Upload
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    loadingState: document.getElementById('loadingState'),
    uploadSection: document.getElementById('uploadSection'),
    dashboardSection: document.getElementById('dashboardSection'),
    
    // Dashboard
    statsGrid: document.getElementById('statsGrid'),
    userTableContainer: document.getElementById('userTableContainer'),
    filterBtn: document.getElementById('filterBtn'),
    filterSection: document.getElementById('filterSection'),
    exportBtn: document.getElementById('exportBtn'),
    searchUser: document.getElementById('searchUser'),
    filterRisk: document.getElementById('filterRisk'),
    
    // Message Testing
    messageInput: document.getElementById('messageInput'),
    charCount: document.getElementById('charCount'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    messageLoadingState: document.getElementById('messageLoadingState'),
    resultsSection: document.getElementById('resultsSection'),
    
    // Results
    predictionBadge: document.getElementById('predictionBadge'),
    urgencyBadge: document.getElementById('urgencyBadge'),
    confidenceValue: document.getElementById('confidenceValue'),
    confidenceBar: document.getElementById('confidenceBar'),
    severityText: document.getElementById('severityText'),
    actionText: document.getElementById('actionText'),
    explanationToggle: document.getElementById('explanationToggle'),
    explanationContent: document.getElementById('explanationContent'),
    explanationDetails: document.getElementById('explanationDetails'),
    
    // Modal
    exportModal: document.getElementById('exportModal'),
    closeModal: document.getElementById('closeModal'),
    exportCSV: document.getElementById('exportCSV'),
    exportJSON: document.getElementById('exportJSON'),
    
    // Theme
    themeToggle: document.getElementById('themeToggle'),
    
    // Toast
    toast: document.getElementById('toast'),
    toastMessage: document.getElementById('toastMessage'),
};

// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    initializeUpload();
    initializeMessageTesting();
    initializeTheme();
    initializeFilters();
    initializeModal();
});

// ==================== TAB SYSTEM ====================
function initializeTabs() {
    elements.tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.dataset.tab;
            switchTab(tabName);
        });
    });
}

function switchTab(tabName) {
    // Update buttons
    elements.tabButtons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    
    // Update content
    elements.tabContents.forEach(content => {
        if (tabName === 'csv' && content.id === 'csvTab') {
            content.classList.add('active');
        } else if (tabName === 'message' && content.id === 'messageTab') {
            content.classList.add('active');
        } else {
            content.classList.remove('active');
        }
    });
}

// ==================== CSV UPLOAD ====================
function initializeUpload() {
    // Click to upload
    elements.uploadArea.addEventListener('click', () => {
        elements.fileInput.click();
    });
    
    // File input change
    elements.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });
    
    // Drag and drop
    elements.uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.add('dragover');
    });
    
    elements.uploadArea.addEventListener('dragleave', () => {
        elements.uploadArea.classList.remove('dragover');
    });
    
    elements.uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
}

async function handleFile(file) {
    // Validate file
    if (!file.name.endsWith('.csv')) {
        showToast('Please upload a CSV file', 'error');
        return;
    }
    
    // Show loading
    elements.uploadArea.style.display = 'none';
    elements.loadingState.classList.add('active');
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        // Upload to API
        const response = await fetch(`${API_URL}/process-csv`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Failed to process CSV');
        }
        
        const data = await response.json();
        analyticsData = data;
        
        // Hide upload, show dashboard
        elements.uploadSection.style.display = 'none';
        elements.dashboardSection.style.display = 'block';
        
        // Render dashboard
        renderDashboard(data);
        
        showToast('CSV processed successfully!', 'success');
        
    } catch (error) {
        console.error('Error processing CSV:', error);
        showToast('Error processing CSV: ' + error.message, 'error');
        
        // Reset UI
        elements.uploadArea.style.display = 'block';
        elements.loadingState.classList.remove('active');
    }
}

// ==================== DASHBOARD RENDERING ====================
function renderDashboard(data) {
    renderStats(data.summary);
    renderCharts(data);
    renderUserTable(data.top_risk_users || []);
    allUsers = data.top_risk_users || [];
}

function renderStats(summary) {
    const stats = [
        { label: 'Total Comments', value: formatNumber(summary.total_comments || 0), class: 'primary' },
        { label: 'Total Users', value: formatNumber(summary.total_users || 0), class: 'secondary' },
        { label: 'Hate Speech %', value: (summary.overall_hate_percentage || 0).toFixed(1) + '%', class: 'danger' },
        { label: 'Offensive %', value: (summary.overall_offensive_percentage || 0).toFixed(1) + '%', class: 'warning' },
        { label: 'Neither %', value: (summary.overall_neither_percentage || 0).toFixed(1) + '%', class: 'success' },
        { label: 'Avg Severity', value: (summary.avg_severity_score || 0).toFixed(2), class: 'secondary' }
    ];
    
    elements.statsGrid.innerHTML = stats.map(stat => `
        <div class="stat-card ${stat.class}">
            <span class="stat-label">${stat.label}</span>
            <div class="stat-value">${stat.value}</div>
        </div>
    `).join('');
}

function renderCharts(data) {
    // Classification Distribution Chart
    const classCtx = document.getElementById('classificationChart');
    if (classCtx) {
        new Chart(classCtx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['Hate Speech', 'Offensive', 'Neither'],
                datasets: [{
                    label: 'Percentage',
                    data: [
                        data.summary.overall_hate_percentage || 0,
                        data.summary.overall_offensive_percentage || 0,
                        data.summary.overall_neither_percentage || 0
                    ],
                    backgroundColor: [
                        'rgba(239, 68, 68, 0.8)',
                        'rgba(245, 158, 11, 0.8)',
                        'rgba(16, 185, 129, 0.8)'
                    ],
                    borderColor: [
                        'rgb(239, 68, 68)',
                        'rgb(245, 158, 11)',
                        'rgb(16, 185, 129)'
                    ],
                    borderWidth: 2,
                    borderRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.9)',
                        padding: 12,
                        borderRadius: 8,
                        titleFont: { size: 14, weight: '600' },
                        bodyFont: { size: 13 }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(100, 116, 139, 0.1)'
                        },
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }
    
    // Risk Levels Chart
    const riskCtx = document.getElementById('riskLevelsChart');
    if (riskCtx) {
        const riskCounts = data.summary.users_by_risk || {};
        new Chart(riskCtx.getContext('2d'), {
            type: 'doughnut',
            data: {
                labels: ['Critical', 'High', 'Medium', 'Low', 'Minimal'],
                datasets: [{
                    data: [
                        riskCounts.CRITICAL || 0,
                        riskCounts.HIGH || 0,
                        riskCounts.MEDIUM || 0,
                        riskCounts.LOW || 0,
                        riskCounts.MINIMAL || 0
                    ],
                    backgroundColor: [
                        'rgba(239, 68, 68, 0.8)',
                        'rgba(245, 158, 11, 0.8)',
                        'rgba(217, 119, 6, 0.8)',
                        'rgba(16, 185, 129, 0.8)',
                        'rgba(100, 116, 139, 0.8)'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 15,
                            font: { size: 13 }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.9)',
                        padding: 12,
                        borderRadius: 8,
                        titleFont: { size: 14, weight: '600' },
                        bodyFont: { size: 13 }
                    }
                }
            }
        });
    }
    
    // Severity Over Time Chart (if available)
    const severityCtx = document.getElementById('severityTimeChart');
    if (severityCtx) {
        // Generate sample time series data
        const timeData = generateTimeSeriesData();
        
        new Chart(severityCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: timeData.labels,
                datasets: [{
                    label: 'Average Severity Score',
                    data: timeData.values,
                    borderColor: 'rgb(37, 99, 235)',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 4,
                    pointHoverRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.9)',
                        padding: 12,
                        borderRadius: 8,
                        titleFont: { size: 14, weight: '600' },
                        bodyFont: { size: 13 }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: {
                            color: 'rgba(100, 116, 139, 0.1)'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }
}

function generateTimeSeriesData() {
    const labels = [];
    const values = [];
    const now = new Date();
    
    for (let i = 20; i >= 0; i--) {
        const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
        labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
        values.push(Math.random() * 60 + 20); // Random values between 20-80
    }
    
    return { labels, values };
}

function renderUserTable(users) {
    if (users.length === 0) {
        elements.userTableContainer.innerHTML = '<p class="text-center">No users found</p>';
        return;
    }
    
    const tableHTML = `
        <table>
            <thead>
                <tr>
                    <th>Username</th>
                    <th>Risk Score</th>
                    <th>Risk Level</th>
                    <th>Hate %</th>
                    <th>Total Comments</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                ${users.map((user, idx) => `
                    <tr>
                        <td><strong>${escapeHtml(user.user)}</strong></td>
                        <td>${user.risk_score.toFixed(2)}</td>
                        <td><span class="badge ${user.risk_level.toLowerCase()}">${user.risk_level}</span></td>
                        <td>${user.hate_percentage.toFixed(1)}%</td>
                        <td>${user.total_comments}</td>
                        <td>
                            <button class="btn btn-secondary" style="padding: 6px 12px; font-size: 13px;" onclick="viewUserDetails('${escapeHtml(user.user)}')">
                                <span class="material-icons" style="font-size: 16px;">visibility</span>
                                View
                            </button>
                        </td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
    
    elements.userTableContainer.innerHTML = tableHTML;
}

function viewUserDetails(username) {
    if (!analyticsData || !analyticsData.user_analytics) return;
    
    const userAnalytics = analyticsData.user_analytics[username];
    if (!userAnalytics) {
        showToast('User details not found', 'error');
        return;
    }
    
    const details = `
        <h3>User Analytics: ${username}</h3>
        <div class="results-grid mt-3">
            <div class="result-item">
                <span class="result-label">Total Comments</span>
                <strong>${userAnalytics.total_comments}</strong>
            </div>
            <div class="result-item">
                <span class="result-label">Hate Speech</span>
                <strong>${userAnalytics.hate_speech} (${userAnalytics.hate_percentage}%)</strong>
            </div>
            <div class="result-item">
                <span class="result-label">Offensive</span>
                <strong>${userAnalytics.offensive} (${userAnalytics.offensive_percentage}%)</strong>
            </div>
            <div class="result-item">
                <span class="result-label">Neither</span>
                <strong>${userAnalytics.neither} (${userAnalytics.neither_percentage}%)</strong>
            </div>
            <div class="result-item">
                <span class="result-label">Average Severity</span>
                <strong>${userAnalytics.avg_severity_score}</strong>
            </div>
            <div class="result-item">
                <span class="result-label">Max Severity</span>
                <strong>${userAnalytics.max_severity_score}</strong>
            </div>
        </div>
        ${userAnalytics.most_severe_comment && userAnalytics.most_severe_comment.text ? `
            <div class="mt-3" style="padding: 16px; background: rgba(239, 68, 68, 0.1); border-left: 4px solid var(--danger-color); border-radius: 8px;">
                <strong>Most Severe Comment:</strong><br>
                <p style="margin-top: 8px;">"${escapeHtml(userAnalytics.most_severe_comment.text)}"</p>
                <small style="color: var(--text-secondary);">Severity: ${userAnalytics.most_severe_comment.severity}</small>
            </div>
        ` : ''}
    `;
    
    // Create and show modal with details
    showCustomModal(details);
}

// ==================== MESSAGE TESTING ====================
function initializeMessageTesting() {
    // Character counter
    elements.messageInput.addEventListener('input', updateCharCount);
    
    // Analyze button
    elements.analyzeBtn.addEventListener('click', analyzeMessage);
    
    // Enter key to analyze (Ctrl+Enter)
    elements.messageInput.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            analyzeMessage();
        }
    });
    
    // Explanation toggle
    elements.explanationToggle.addEventListener('click', toggleExplanation);
}

function updateCharCount() {
    const length = elements.messageInput.value.length;
    elements.charCount.textContent = `${length}/5000 characters`;
}

async function analyzeMessage() {
    const text = elements.messageInput.value.trim();
    
    if (!text) {
        showToast('Please enter a message to analyze', 'error');
        return;
    }
    
    // Show loading
    elements.analyzeBtn.disabled = true;
    elements.messageLoadingState.style.display = 'block';
    elements.resultsSection.style.display = 'none';
    
    try {
        const response = await fetch(`${API_URL}/classify`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                include_severity: true,
                include_explanation: true
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to analyze message');
        }
        
        const result = await response.json();
        
        // Hide loading, show results
        elements.messageLoadingState.style.display = 'none';
        elements.resultsSection.style.display = 'block';
        
        // Render results
        renderResults(result);
        
        // Scroll to results
        elements.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        
    } catch (error) {
        console.error('Error analyzing message:', error);
        showToast('Error analyzing message: ' + error.message, 'error');
        elements.messageLoadingState.style.display = 'none';
    } finally {
        elements.analyzeBtn.disabled = false;
    }
}

function renderResults(result) {
    // Prediction badge
    const predictionClass = result.prediction.toLowerCase().replace(' ', '-');
    elements.predictionBadge.className = `badge ${predictionClass}`;
    elements.predictionBadge.textContent = result.prediction;
    
    // Urgency badge
    if (result.action && result.action.urgency) {
        const urgencyClass = result.action.urgency.toLowerCase();
        elements.urgencyBadge.className = `badge ${urgencyClass}`;
        elements.urgencyBadge.textContent = result.action.urgency;
    } else {
        elements.urgencyBadge.className = 'badge minimal';
        elements.urgencyBadge.textContent = 'NONE';
    }
    
    // Confidence
    const confidence = Math.round(result.confidence * 100);
    elements.confidenceValue.textContent = confidence + '%';
    elements.confidenceBar.style.width = confidence + '%';
    
    // Severity
    if (result.severity) {
        const severityLevel = result.severity.level || 1;
        const severityLabel = result.severity.label || 'LOW';
        const severityScore = result.severity.score || 0;
        elements.severityText.textContent = `${severityLabel} (${severityLevel}/5) - Score: ${severityScore}/100`;
    } else {
        elements.severityText.textContent = 'Not available';
    }
    
    // Action
    if (result.action && result.action.description) {
        elements.actionText.textContent = result.action.description;
    } else if (result.action && result.action.primary) {
        elements.actionText.textContent = result.action.primary;
    } else {
        elements.actionText.textContent = 'No action required.';
    }
    
    // Explanation
    if (result.explanation) {
        renderExplanation(result.explanation, result.severity);
    } else {
        elements.explanationDetails.innerHTML = '<p>No explanation available.</p>';
    }
}

function renderExplanation(explanation, severity) {
    let html = '';
    
    // Methods used
    if (explanation.methods_used && explanation.methods_used.length > 0) {
        html += `
            <div class="mb-2">
                <strong>Analysis Methods:</strong>
                <p>${explanation.methods_used.join(', ')}</p>
            </div>
        `;
    }
    
    // Summary
    if (explanation.summary) {
        html += `
            <div class="mb-2">
                <strong>Summary:</strong>
                <p>${escapeHtml(explanation.summary)}</p>
            </div>
        `;
    }
    
    // Severity explanation
    if (severity && severity.explanation) {
        html += `
            <div class="mb-2">
                <strong>Severity Analysis:</strong>
                <p>${escapeHtml(severity.explanation)}</p>
            </div>
        `;
    }
    
    if (!html) {
        html = '<p>No detailed explanation available.</p>';
    }
    
    elements.explanationDetails.innerHTML = html;
}

function toggleExplanation() {
    const isVisible = elements.explanationContent.style.display !== 'none';
    elements.explanationContent.style.display = isVisible ? 'none' : 'block';
    elements.explanationToggle.classList.toggle('active', !isVisible);
}

// ==================== FILTERS ====================
function initializeFilters() {
    elements.filterBtn.addEventListener('click', () => {
        const isVisible = elements.filterSection.style.display !== 'none';
        elements.filterSection.style.display = isVisible ? 'none' : 'flex';
    });
    
    elements.searchUser.addEventListener('input', applyFilters);
    elements.filterRisk.addEventListener('change', applyFilters);
}

function applyFilters() {
    const searchTerm = elements.searchUser.value.toLowerCase();
    const riskLevel = elements.filterRisk.value;
    
    let filtered = allUsers.filter(user => {
        const matchesSearch = user.user.toLowerCase().includes(searchTerm);
        const matchesRisk = !riskLevel || user.risk_level === riskLevel;
        return matchesSearch && matchesRisk;
    });
    
    renderUserTable(filtered);
}

// ==================== EXPORT ====================
function initializeModal() {
    elements.exportBtn.addEventListener('click', () => {
        elements.exportModal.classList.add('active');
    });
    
    elements.closeModal.addEventListener('click', () => {
        elements.exportModal.classList.remove('active');
    });
    
    elements.exportModal.addEventListener('click', (e) => {
        if (e.target === elements.exportModal) {
            elements.exportModal.classList.remove('active');
        }
    });
    
    elements.exportCSV.addEventListener('click', exportToCSV);
    elements.exportJSON.addEventListener('click', exportToJSON);
}

function exportToCSV() {
    if (!allUsers || allUsers.length === 0) {
        showToast('No data to export', 'error');
        return;
    }
    
    const csv = [
        ['User', 'Risk Score', 'Risk Level', 'Hate %', 'Total Comments'],
        ...allUsers.map(u => [
            u.user,
            u.risk_score.toFixed(2),
            u.risk_level,
            u.hate_percentage.toFixed(1),
            u.total_comments
        ])
    ].map(row => row.join(',')).join('\n');
    
    downloadFile('user_analytics.csv', csv, 'text/csv');
    elements.exportModal.classList.remove('active');
    showToast('CSV exported successfully!', 'success');
}

function exportToJSON() {
    if (!analyticsData) {
        showToast('No data to export', 'error');
        return;
    }
    
    const json = JSON.stringify(analyticsData, null, 2);
    downloadFile('analytics_data.json', json, 'application/json');
    elements.exportModal.classList.remove('active');
    showToast('JSON exported successfully!', 'success');
}

function downloadFile(filename, content, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// ==================== THEME ====================
function initializeTheme() {
    // Load saved theme
    const savedTheme = localStorage.getItem('theme') || 'light';
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-mode');
        elements.themeToggle.querySelector('.material-icons').textContent = 'light_mode';
    }
    
    // Toggle theme
    elements.themeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        const isDark = document.body.classList.contains('dark-mode');
        elements.themeToggle.querySelector('.material-icons').textContent = isDark ? 'light_mode' : 'dark_mode';
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
    });
}

// ==================== UTILITIES ====================
function showToast(message, type = 'info') {
    elements.toastMessage.textContent = message;
    elements.toast.classList.add('show');
    
    setTimeout(() => {
        elements.toast.classList.remove('show');
    }, 3000);
}

function showCustomModal(content) {
    const modal = document.createElement('div');
    modal.className = 'modal active';
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h3>User Details</h3>
                <button class="modal-close" onclick="this.closest('.modal').remove()">
                    <span class="material-icons">close</span>
                </button>
            </div>
            <div class="modal-body">
                ${content}
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Close on outside click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.remove();
        }
    });
}

function formatNumber(num) {
    return num.toLocaleString();
}

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// ==================== EXPOSE FUNCTIONS ====================
window.viewUserDetails = viewUserDetails;