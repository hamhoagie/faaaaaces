/**
 * Unified Mask Detection & Face Reconstruction Workflow
 */

// Global state
let currentStep = 1;
let selectedVideo = null;
let detectionResults = null;
let selectedFaces = [];
let reconstructionResults = null;

// Initialize the workflow
document.addEventListener('DOMContentLoaded', function() {
    initializeWorkflow();
    setupEventListeners();
    loadExistingVideos();
});

function initializeWorkflow() {
    // Show only the first step initially
    for (let i = 2; i <= 5; i++) {
        document.getElementById(`step-${i}`).style.display = 'none';
    }
    updateProgressIndicator(1);
}

function setupEventListeners() {
    // Detection mode radio buttons
    document.getElementById('mode-targeted').addEventListener('change', function() {
        document.getElementById('timestamp-inputs').style.display = this.checked ? 'block' : 'none';
    });
    
    document.getElementById('mode-full').addEventListener('change', function() {
        document.getElementById('timestamp-inputs').style.display = 'none';
    });

    // File upload
    const fileInput = document.getElementById('video-file-input');
    const uploadZone = document.getElementById('upload-zone');

    fileInput.addEventListener('change', handleFileUpload);

    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload({ target: { files } });
        }
    });

    uploadZone.addEventListener('click', () => {
        fileInput.click();
    });
}

// Step 1: Video Selection/Upload
async function loadExistingVideos() {
    try {
        const response = await fetch('/api/videos_list');
        const data = await response.json();
        const videos = data.videos || [];
        
        const videoSelect = document.getElementById('video-select');
        videoSelect.innerHTML = '<option value="">Select a video...</option>';
        
        videos.forEach(video => {
            const option = document.createElement('option');
            option.value = video.id;
            option.textContent = `Video ${video.id}: ${video.filename}`;
            option.dataset.filename = video.filename;
            videoSelect.appendChild(option);
        });
        
        console.log(`Loaded ${videos.length} videos`);
        
    } catch (error) {
        console.error('Error loading videos:', error);
        showAlert('Error loading videos. Please refresh the page.', 'danger');
    }
}

function selectExistingVideo() {
    const videoSelect = document.getElementById('video-select');
    const videoId = videoSelect.value;
    
    if (!videoId) {
        showAlert('Please select a video first.', 'warning');
        return;
    }
    
    const filename = videoSelect.options[videoSelect.selectedIndex].dataset.filename;
    
    selectedVideo = {
        id: videoId,
        filename: filename,
        source: 'existing'
    };
    
    proceedToStep2();
}

async function handleFileUpload(event) {
    const files = event.target.files;
    if (files.length === 0) return;
    
    const file = files[0];
    
    // Validate file type
    if (!file.type.startsWith('video/')) {
        showAlert('Please select a valid video file.', 'danger');
        return;
    }
    
    // Show upload progress
    showAlert('Uploading video...', 'info');
    
    try {
        const formData = new FormData();
        formData.append('video', file);
        
        const response = await fetch('/api/upload_video', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Upload failed: ${response.status}`);
        }
        
        const result = await response.json();
        
        selectedVideo = {
            id: result.video_id,
            filename: file.name,
            source: 'upload'
        };
        
        showAlert('Video uploaded successfully!', 'success');
        proceedToStep2();
        
    } catch (error) {
        console.error('Upload error:', error);
        showAlert(`Upload failed: ${error.message}`, 'danger');
    }
}

async function processVideoUrl() {
    const url = document.getElementById('video-url').value.trim();
    
    if (!url) {
        showAlert('Please enter a video URL.', 'warning');
        return;
    }
    
    showAlert('Downloading video from URL...', 'info');
    
    try {
        const response = await fetch('/api/process_url', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ url })
        });
        
        if (!response.ok) {
            throw new Error(`Download failed: ${response.status}`);
        }
        
        const result = await response.json();
        
        selectedVideo = {
            id: result.video_id,
            filename: `Video from URL`,
            source: 'url'
        };
        
        showAlert('Video downloaded successfully!', 'success');
        proceedToStep2();
        
    } catch (error) {
        console.error('URL processing error:', error);
        showAlert(`Download failed: ${error.message}`, 'danger');
    }
}

function proceedToStep2() {
    // Update video info
    document.getElementById('current-video-name').textContent = selectedVideo.filename;
    document.getElementById('selected-video-info').style.display = 'block';
    
    // Enable detection button
    document.getElementById('start-detection-btn').disabled = false;
    
    // Move to step 2
    moveToStep(2);
}

// Step 2: Mask Detection
async function startDetection() {
    if (!selectedVideo) {
        showAlert('No video selected.', 'danger');
        return;
    }
    
    const detectionMode = document.querySelector('input[name="detection-mode"]:checked').value;
    
    // Show progress
    document.getElementById('detection-progress').style.display = 'block';
    document.getElementById('start-detection-btn').disabled = true;
    
    try {
        updateProgress('detection', 10, 'Initializing mask detection...');
        
        let response;
        
        if (detectionMode === 'full') {
            updateProgress('detection', 30, 'Analyzing full video...');
            response = await fetch(`/api/enhanced/enhanced_detect_masks/${selectedVideo.id}`, {
                method: 'POST'
            });
        } else {
            const timestamps = getTimestamps();
            if (timestamps.length === 0) {
                showAlert('Please enter at least one timestamp.', 'warning');
                resetDetectionProgress();
                return;
            }
            
            updateProgress('detection', 30, 'Processing targeted timestamps...');
            response = await fetch(`/api/enhanced/enhanced_extract_timestamps/${selectedVideo.id}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ timestamps })
            });
        }
        
        updateProgress('detection', 70, 'Analyzing faces and masks...');
        
        if (!response.ok) {
            throw new Error(`Detection failed: ${response.status}`);
        }
        
        const results = await response.json();
        detectionResults = results;
        
        updateProgress('detection', 100, 'Detection complete!');
        
        // Load all masked faces to get comprehensive results
        await loadAllMaskedFaces();
        
        proceedToStep3();
        
    } catch (error) {
        console.error('Detection error:', error);
        showAlert(`Detection failed: ${error.message}`, 'danger');
        resetDetectionProgress();
    }
}

async function loadAllMaskedFaces() {
    try {
        const response = await fetch('/api/masked/all_masked_faces');
        const data = await response.json();
        
        // Filter faces for current video
        const videoFaces = data.all_faces.filter(face => face.video_id == selectedVideo.id);
        
        detectionResults = {
            total_faces: videoFaces.length,
            masked_faces: videoFaces.filter(face => face.is_enhanced_detection).length,
            unmasked_faces: videoFaces.filter(face => !face.is_enhanced_detection).length,
            results: videoFaces.map(face => ({
                ...face,
                is_masked: face.is_enhanced_detection,
                confidence: 0.8 // Default confidence
            }))
        };
        
    } catch (error) {
        console.error('Error loading masked faces:', error);
    }
}

function getTimestamps() {
    const inputs = document.querySelectorAll('#timestamp-container input[type="number"]');
    return Array.from(inputs)
        .map(input => parseFloat(input.value))
        .filter(val => !isNaN(val) && val >= 0);
}

function addTimestamp() {
    const container = document.getElementById('timestamp-container');
    const input = document.createElement('input');
    input.type = 'number';
    input.className = 'form-control mb-2';
    input.step = '0.1';
    input.placeholder = 'Time in seconds';
    container.appendChild(input);
}

function proceedToStep3() {
    // Update detection results
    document.getElementById('total-faces-count').textContent = detectionResults.total_faces || 0;
    document.getElementById('masked-faces-count').textContent = detectionResults.masked_faces || 0;
    document.getElementById('unmasked-faces-count').textContent = detectionResults.unmasked_faces || 0;
    
    const maskRate = detectionResults.total_faces > 0 
        ? Math.round((detectionResults.masked_faces / detectionResults.total_faces) * 100)
        : 0;
    document.getElementById('mask-detection-rate').textContent = maskRate + '%';
    
    // Display faces
    displayFaces(detectionResults.results || []);
    
    // Show results
    document.getElementById('detection-results').style.display = 'block';
    
    moveToStep(3);
}

// Step 3: Results Review
function displayFaces(faces) {
    const grid = document.getElementById('faces-grid');
    grid.innerHTML = '';
    
    if (faces.length === 0) {
        grid.innerHTML = '<p class="text-muted text-center col-12">No faces detected.</p>';
        return;
    }
    
    faces.forEach((face, index) => {
        const faceCard = createFaceCard(face, index);
        grid.appendChild(faceCard);
    });
}

function createFaceCard(face, index) {
    const card = document.createElement('div');
    const maskStatus = face.is_masked ? 'masked' : 'unmasked';
    card.className = `face-card ${maskStatus}`;
    card.dataset.index = index;
    card.dataset.maskStatus = maskStatus;
    
    const imagePath = face.image_path || `/faces/${face.directory}/${face.filename}`;
    
    card.innerHTML = `
        <img src="${imagePath}" alt="Face ${index + 1}" class="face-preview" 
             onclick="openImageModal('${imagePath}', '${maskStatus === 'masked' ? 'Masked' : 'Unmasked'} Face - ${face.timestamp.toFixed(1)}s')"
             style="cursor: pointer;"
             onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgZmlsbD0iI2VlZSIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LXNpemU9IjE0IiBmaWxsPSIjOTk5IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+Tm8gSW1hZ2U8L3RleHQ+PC9zdmc+'">
        <div class="face-info">
            <div class="badge ${maskStatus === 'masked' ? 'bg-danger' : 'bg-success'} mb-2">
                ${maskStatus.toUpperCase()}
            </div>
            <div class="small text-muted">
                ${face.timestamp.toFixed(1)}s
            </div>
            ${face.mask_type ? `<div class="small text-muted">${face.mask_type}</div>` : ''}
        </div>
        ${maskStatus === 'masked' ? 
            `<div class="form-check mt-2">
                <input class="form-check-input" type="checkbox" onchange="toggleFaceSelection(${index})">
                <label class="form-check-label small">Select for reconstruction</label>
            </div>` : ''
        }
    `;
    
    return card;
}

function toggleFaceSelection(index) {
    const checkbox = document.querySelector(`.face-card[data-index="${index}"] input[type="checkbox"]`);
    const card = document.querySelector(`.face-card[data-index="${index}"]`);
    
    if (checkbox.checked) {
        selectedFaces.push(index);
        card.classList.add('selected');
    } else {
        selectedFaces = selectedFaces.filter(i => i !== index);
        card.classList.remove('selected');
    }
    
    updateSelectedCount();
}

function filterFaces(filter) {
    const cards = document.querySelectorAll('.face-card');
    cards.forEach(card => {
        const maskStatus = card.dataset.maskStatus;
        if (filter === 'all' || maskStatus === filter) {
            card.style.display = 'block';
        } else {
            card.style.display = 'none';
        }
    });
}

function updateSelectedCount() {
    document.getElementById('selected-count').textContent = selectedFaces.length;
    document.getElementById('continue-to-reconstruction').disabled = selectedFaces.length === 0;
}

function continueToReconstruction() {
    if (selectedFaces.length === 0) {
        showAlert('Please select at least one masked face for reconstruction.', 'warning');
        return;
    }
    
    // Show selected faces preview
    showSelectedFacesPreview();
    
    moveToStep(4);
}

// Step 4: Face Reconstruction
function showSelectedFacesPreview() {
    const container = document.getElementById('selected-faces-preview');
    const maskedFaces = detectionResults.results.filter(face => face.is_masked);
    const selectedFacesData = selectedFaces.map(index => maskedFaces[index]);
    
    container.innerHTML = `
        <p class="mb-3">${selectedFaces.length} faces selected for reconstruction:</p>
        <div class="row">
            ${selectedFacesData.map((face, index) => `
                <div class="col-md-6 mb-2">
                    <div class="d-flex align-items-center">
                        <img src="${face.image_path || `/faces/${face.directory}/${face.filename}`}" 
                             style="width: 40px; height: 40px; object-fit: cover; border-radius: 6px;">
                        <div class="ms-2">
                            <small>${face.timestamp.toFixed(1)}s</small>
                        </div>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
    
    document.getElementById('reconstruction-setup').style.display = 'block';
}

async function startReconstruction() {
    const method = document.getElementById('reconstruction-method').value;
    const saveOriginals = document.getElementById('save-originals').checked;
    
    // Show progress
    document.getElementById('reconstruction-progress').style.display = 'block';
    document.getElementById('start-reconstruction-btn').disabled = true;
    
    try {
        updateProgress('reconstruction', 20, 'Preparing faces for reconstruction...');
        
        const maskedFaces = detectionResults.results.filter(face => face.is_masked);
        const selectedFacesData = selectedFaces.map(index => maskedFaces[index]);
        const faceIds = selectedFacesData.map(face => face.face_id);
        
        updateProgress('reconstruction', 50, 'Reconstructing faces...');
        
        const response = await fetch('/api/masked/batch_reconstruct', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                face_ids: faceIds,
                method: method,
                save_originals: saveOriginals
            })
        });
        
        if (!response.ok) {
            throw new Error(`Reconstruction failed: ${response.status}`);
        }
        
        const results = await response.json();
        reconstructionResults = results;
        
        updateProgress('reconstruction', 100, 'Reconstruction complete!');
        
        proceedToStep5();
        
    } catch (error) {
        console.error('Reconstruction error:', error);
        showAlert(`Reconstruction failed: ${error.message}`, 'danger');
        resetReconstructionProgress();
    }
}

// Step 5: Final Results
function proceedToStep5() {
    displayReconstructionResults();
    moveToStep(5);
}

function displayReconstructionResults() {
    const grid = document.getElementById('reconstruction-grid');
    grid.innerHTML = '';
    
    if (!reconstructionResults || !reconstructionResults.results) {
        grid.innerHTML = '<p class="text-muted text-center">No reconstruction results available.</p>';
        return;
    }
    
    const maskedFaces = detectionResults.results.filter(face => face.is_masked);
    
    reconstructionResults.results.forEach((result, index) => {
        const originalFace = maskedFaces.find(face => face.face_id === result.face_id);
        if (!originalFace) return;
        
        const card = document.createElement('div');
        card.className = 'reconstruction-card';
        
        card.innerHTML = `
            <h6 class="mb-3">Face at ${originalFace.timestamp.toFixed(1)}s</h6>
            <div class="comparison-container">
                <div class="comparison-item">
                    <h6>Original (Masked)</h6>
                    <img src="${originalFace.image_path || `/faces/${originalFace.directory}/${originalFace.filename}`}" 
                         alt="Original masked face"
                         onclick="openImageModal('${originalFace.image_path || `/faces/${originalFace.directory}/${originalFace.filename}`}', 'Original Masked Face - ${originalFace.timestamp.toFixed(1)}s')"
                         style="cursor: pointer; border: 2px solid transparent; border-radius: 8px; transition: border-color 0.3s;"
                         onmouseover="this.style.borderColor='#0d6efd'"
                         onmouseout="this.style.borderColor='transparent'">
                    <div class="mt-2">
                        <small class="text-muted">Original</small>
                    </div>
                </div>
                <div class="comparison-item">
                    <h6>Reconstructed</h6>
                    <img src="${result.reconstructed_path}" alt="Reconstructed face"
                         onclick="openImageModal('${result.reconstructed_path}', 'Reconstructed Face - ${originalFace.timestamp.toFixed(1)}s')"
                         style="cursor: pointer; border: 2px solid transparent; border-radius: 8px; transition: border-color 0.3s;"
                         onmouseover="this.style.borderColor='#198754'"
                         onmouseout="this.style.borderColor='transparent'">
                    <div class="mt-2">
                        <small class="text-muted">
                            Quality: ${(result.quality_score * 100).toFixed(0)}%<br>
                            Method: ${result.method}
                        </small>
                    </div>
                </div>
            </div>
            <div class="text-center mt-3">
                <button class="btn btn-outline-primary btn-sm" onclick="downloadReconstruction('${result.face_id}')">
                    <i class="fas fa-download"></i> Download
                </button>
            </div>
        `;
        
        grid.appendChild(card);
    });
    
    document.getElementById('final-results').style.display = 'block';
}

// Utility Functions
function moveToStep(stepNumber) {
    // Hide current step
    document.getElementById(`step-${currentStep}`).style.display = 'none';
    document.getElementById(`step-${currentStep}`).classList.remove('active');
    
    // Show new step
    document.getElementById(`step-${stepNumber}`).style.display = 'block';
    document.getElementById(`step-${stepNumber}`).classList.add('active');
    
    // Update progress
    updateProgressIndicator(stepNumber);
    
    currentStep = stepNumber;
}

function updateProgressIndicator(stepNumber) {
    for (let i = 1; i <= 5; i++) {
        const indicator = document.getElementById(`progress-step-${i}`);
        indicator.classList.remove('active', 'completed');
        
        if (i < stepNumber) {
            indicator.classList.add('completed');
        } else if (i === stepNumber) {
            indicator.classList.add('active');
        }
    }
}

function updateProgress(type, percentage, message) {
    const progressBar = document.getElementById(`${type}-progress-bar`);
    const statusText = document.getElementById(`${type}-status`);
    
    if (progressBar) {
        progressBar.style.width = percentage + '%';
    }
    
    if (statusText) {
        statusText.textContent = message || `${percentage}%`;
    }
}

function resetDetectionProgress() {
    document.getElementById('detection-progress').style.display = 'none';
    document.getElementById('start-detection-btn').disabled = false;
}

function resetReconstructionProgress() {
    document.getElementById('reconstruction-progress').style.display = 'none';
    document.getElementById('start-reconstruction-btn').disabled = false;
}

function resetWorkflow() {
    // Reset all state
    currentStep = 1;
    selectedVideo = null;
    detectionResults = null;
    selectedFaces = [];
    reconstructionResults = null;
    
    // Reset UI
    initializeWorkflow();
    
    // Clear forms
    document.getElementById('video-select').value = '';
    document.getElementById('video-url').value = '';
    document.getElementById('video-file-input').value = '';
    
    // Hide progress sections
    document.getElementById('detection-progress').style.display = 'none';
    document.getElementById('reconstruction-progress').style.display = 'none';
    document.getElementById('selected-video-info').style.display = 'none';
    document.getElementById('detection-results').style.display = 'none';
    document.getElementById('reconstruction-setup').style.display = 'none';
    document.getElementById('final-results').style.display = 'none';
    
    // Reset buttons
    document.getElementById('start-detection-btn').disabled = true;
    document.getElementById('continue-to-reconstruction').disabled = true;
}

function downloadReconstruction(faceId) {
    const result = reconstructionResults.results.find(r => r.face_id === faceId);
    if (result && result.reconstructed_path) {
        const link = document.createElement('a');
        link.href = result.reconstructed_path;
        link.download = `reconstructed_${faceId}.jpg`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
}

function downloadAllResults() {
    if (reconstructionResults && reconstructionResults.results) {
        reconstructionResults.results.forEach(result => {
            setTimeout(() => downloadReconstruction(result.face_id), 100);
        });
    }
}

function exportResults() {
    // Create a summary report
    const report = {
        video: selectedVideo,
        detection: {
            total_faces: detectionResults.total_faces,
            masked_faces: detectionResults.masked_faces,
            unmasked_faces: detectionResults.unmasked_faces
        },
        reconstruction: {
            method: document.getElementById('reconstruction-method').value,
            faces_reconstructed: reconstructionResults.successful_reconstructions,
            results: reconstructionResults.results
        },
        timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `mask_analysis_report_${selectedVideo.id}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

function openImageModal(imageSrc, title) {
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('imageModalImg');
    const modalTitle = document.getElementById('imageModalTitle');
    
    modalImg.src = imageSrc;
    modalTitle.textContent = title;
    
    // Store current image URL for download
    modal.dataset.currentImageSrc = imageSrc;
    
    const bootstrapModal = new bootstrap.Modal(modal);
    bootstrapModal.show();
}

function downloadModalImage() {
    const modal = document.getElementById('imageModal');
    const imageSrc = modal.dataset.currentImageSrc;
    
    if (imageSrc) {
        const link = document.createElement('a');
        link.href = imageSrc;
        link.download = imageSrc.split('/').pop() || 'face_image.jpg';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
}

function showAlert(message, type = 'info') {
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const main = document.querySelector('main .workflow-container');
    main.insertBefore(alert, main.firstChild);
    
    setTimeout(() => {
        if (alert.parentNode) {
            alert.remove();
        }
    }, 5000);
}