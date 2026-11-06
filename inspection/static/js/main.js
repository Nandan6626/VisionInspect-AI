// VisionInspect AI - Frontend Scripts

document.addEventListener('DOMContentLoaded', function () {
    // File upload elements
    const dropzone = document.getElementById('uploadDropzone');
    const fileInput = document.getElementById('componentImageInput');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const fileNameSpan = document.getElementById('fileName');
    const fileSizeSpan = document.getElementById('fileSize');
    const removeBtn = document.getElementById('removeImageBtn');
    const inspectForm = document.getElementById('inspectForm');
    const inspectSubmitBtn = document.getElementById('inspectSubmitBtn');
    const loadingOverlay = document.getElementById('loadingOverlay');

    if (dropzone && fileInput) {
        // Dropzone click opens file browser
        dropzone.addEventListener('click', (e) => {
            if (e.target !== fileInput) {
                fileInput.click();
            }
        });

        // Drag & Drop events
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove('dragover');
            });
        });

        dropzone.addEventListener('drop', (e) => {
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                handleFileSelect(fileInput.files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files && fileInput.files.length > 0) {
                handleFileSelect(fileInput.files[0]);
            }
        });
    }

    function handleFileSelect(file) {
        if (!file) return;

        // Verify format
        const validExtensions = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp'];
        if (!validExtensions.includes(file.type)) {
            alert('Please select a valid image file (JPG, JPEG, PNG, WEBP).');
            return;
        }

        // Display metadata
        if (fileNameSpan) fileNameSpan.textContent = file.name;
        if (fileSizeSpan) fileSizeSpan.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';

        // Render preview
        const reader = new FileReader();
        reader.onload = function (e) {
            if (previewImage) previewImage.src = e.target.result;
            if (dropzone) dropzone.style.display = 'none';
            if (previewContainer) previewContainer.style.display = 'block';
            if (inspectSubmitBtn) inspectSubmitBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    if (removeBtn) {
        removeBtn.addEventListener('click', () => {
            if (fileInput) fileInput.value = '';
            if (previewContainer) previewContainer.style.display = 'none';
            if (dropzone) dropzone.style.display = 'block';
            if (previewImage) previewImage.src = '';
            if (inspectSubmitBtn) inspectSubmitBtn.disabled = true;
        });
    }

    if (inspectForm) {
        inspectForm.addEventListener('submit', function (e) {
            if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
                e.preventDefault();
                alert('Please upload an image before proceeding.');
                return;
            }

            // Disable submit button and show loading state
            if (inspectSubmitBtn) {
                inspectSubmitBtn.disabled = true;
                inspectSubmitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status"></span>Analyzing Component...';
            }

            // Show full screen overlay
            if (loadingOverlay) {
                loadingOverlay.style.display = 'flex';
            }
        });
    }
});

