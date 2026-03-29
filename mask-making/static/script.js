const bgCanvas = document.getElementById('bgCanvas');
const maskCanvas = document.getElementById('maskCanvas');
const tempCanvas = document.getElementById('tempCanvas');
const bgCtx = bgCanvas.getContext('2d');
const maskCtx = maskCanvas.getContext('2d');
const tempCtx = tempCanvas.getContext('2d');

const imageUpload = document.getElementById('imageUpload');
const dropZone = document.getElementById('dropZone');
const canvasWrapper = document.getElementById('canvasWrapper');
const brushSizeInput = document.getElementById('brushSize');
const grayLevelInput = document.getElementById('grayLevel');
const grayPreview = document.getElementById('grayPreview');
const downloadBtn = document.getElementById('downloadBtn');
const clearBtn = document.getElementById('clearBtn');
const undoBtn = document.getElementById('undoBtn');
const sizeValue = document.getElementById('sizeValue');
const grayValue = document.getElementById('grayValue');
const imageDim = document.getElementById('imageDim');
const toolBtns = document.querySelectorAll('.tool-btn');

let currentTool = 'brush';
let isDrawing = false;
let startX, startY;
let currentFile = null;
let originalWidth, originalHeight;
let undoStack = [];
const MAX_UNDO = 10;

// Initialize
grayPreview.style.backgroundColor = `rgb(255, 255, 255)`;

// Event Listeners
imageUpload.addEventListener('change', (e) => handleFile(e.target.files[0]));

dropZone.addEventListener('click', () => imageUpload.click());
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.borderColor = 'var(--accent-color)'; });
dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = 'var(--border)'; });
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    handleFile(e.dataTransfer.files[0]);
});

toolBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        toolBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentTool = btn.dataset.tool;
    });
});

brushSizeInput.addEventListener('input', () => {
    sizeValue.textContent = brushSizeInput.value;
});

grayLevelInput.addEventListener('input', () => {
    const val = grayLevelInput.value;
    grayValue.textContent = val;
    grayPreview.style.backgroundColor = `rgb(${val}, ${val}, ${val})`;
});

function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return;

    showToast("Processing image...");
    const formData = new FormData();
    formData.append('image', file);

    fetch('/upload', { method: 'POST', body: formData })
        .then(res => res.json())
        .then(data => {
            currentFile = data.filename;
            originalWidth = data.width;
            originalHeight = data.height;
            imageDim.textContent = `${originalWidth} x ${originalHeight} px`;

            const img = new Image();
            img.onload = () => {
                setupCanvas(img);
                dropZone.classList.add('hide');
                downloadBtn.disabled = false;
                saveState();
                showToast("Image loaded successfully!");
            };
            img.src = data.url;
        });
}

function setupCanvas(img) {
    // Set actual resolution
    bgCanvas.width = maskCanvas.width = tempCanvas.width = originalWidth;
    bgCanvas.height = maskCanvas.height = tempCanvas.height = originalHeight;

    // Scale display based on screen size
    const maxWidth = window.innerWidth * 0.7;
    const maxHeight = window.innerHeight * 0.7;
    let displayWidth = originalWidth;
    let displayHeight = originalHeight;

    if (displayWidth > maxWidth) {
        displayHeight *= maxWidth / displayWidth;
        displayWidth = maxWidth;
    }
    if (displayHeight > maxHeight) {
        displayWidth *= maxHeight / displayHeight;
        displayHeight = maxHeight;
    }

    canvasWrapper.style.width = displayWidth + 'px';
    canvasWrapper.style.height = displayHeight + 'px';

    bgCanvas.style.width = maskCanvas.style.width = tempCanvas.style.width = '100%';
    bgCanvas.style.height = maskCanvas.style.height = tempCanvas.style.height = '100%';

    bgCtx.drawImage(img, 0, 0);

    // Fill mask with transparent black initially
    maskCtx.clearRect(0, 0, originalWidth, originalHeight);
    tempCtx.clearRect(0, 0, originalWidth, originalHeight);
}

// Drawing Logic
tempCanvas.addEventListener('mousedown', startDrawing);
tempCanvas.addEventListener('mousemove', draw);
tempCanvas.addEventListener('mouseup', stopDrawing);
tempCanvas.addEventListener('mouseout', stopDrawing);

function getPos(e) {
    const rect = tempCanvas.getBoundingClientRect();
    const scaleX = originalWidth / rect.width;
    const scaleY = originalHeight / rect.height;
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

function startDrawing(e) {
    if (!currentFile) return;
    isDrawing = true;
    const pos = getPos(e);
    startX = pos.x;
    startY = pos.y;

    if (currentTool === 'brush' || currentTool === 'eraser') {
        saveState();
        draw(e);
    }
}

function draw(e) {
    if (!isDrawing) return;
    const pos = getPos(e);
    const canvas = (currentTool === 'brush' || currentTool === 'eraser') ? maskCanvas : tempCanvas;
    const ctx = (currentTool === 'brush' || currentTool === 'eraser') ? maskCtx : tempCtx;

    const size = parseInt(brushSizeInput.value);
    const gray = parseInt(grayLevelInput.value);

    if (currentTool === 'brush' || currentTool === 'eraser') {
        ctx.beginPath();
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.lineWidth = size;

        if (currentTool === 'eraser') {
            ctx.globalCompositeOperation = 'destination-out';
        } else {
            ctx.globalCompositeOperation = 'source-over';
            ctx.strokeStyle = `rgb(${gray}, ${gray}, ${gray})`;
        }

        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
    } else {
        // Shapes on temp canvas
        tempCtx.clearRect(0, 0, originalWidth, originalHeight);
        tempCtx.fillStyle = `rgb(${gray}, ${gray}, ${gray})`;

        if (currentTool === 'rectangle') {
            const w = pos.x - startX;
            const h = pos.y - startY;
            tempCtx.fillRect(startX, startY, w, h);
        } else if (currentTool === 'circle') {
            const radius = Math.sqrt(Math.pow(pos.x - startX, 2) + Math.pow(pos.y - startY, 2));
            tempCtx.beginPath();
            tempCtx.arc(startX, startY, radius, 0, Math.PI * 2);
            tempCtx.fill();
        }
    }
}

function stopDrawing() {
    if (!isDrawing) return;
    isDrawing = false;

    if (currentTool === 'rectangle' || currentTool === 'circle') {
        saveState();
        maskCtx.globalCompositeOperation = 'source-over';
        maskCtx.drawImage(tempCanvas, 0, 0);
        tempCtx.clearRect(0, 0, originalWidth, originalHeight);
    }

    maskCtx.beginPath();
}

function saveState() {
    if (undoStack.length >= MAX_UNDO) undoStack.shift();
    undoStack.push(maskCanvas.toDataURL());
}

undoBtn.addEventListener('click', () => {
    if (undoStack.length > 0) {
        const state = undoStack.pop();
        const img = new Image();
        img.onload = () => {
            maskCtx.clearRect(0, 0, originalWidth, originalHeight);
            maskCtx.globalCompositeOperation = 'source-over';
            maskCtx.drawImage(img, 0, 0);
        };
        img.src = state;
    }
});

clearBtn.addEventListener('click', () => {
    if (confirm("Clear all mask data?")) {
        saveState();
        maskCtx.clearRect(0, 0, originalWidth, originalHeight);
    }
});

downloadBtn.addEventListener('click', () => {
    showToast("Processing mask for download...");

    // Create a final canvas that for sure has no transparency where we drew
    // and is 1-channel style if possible, but backend will handle L conversion.
    const dataURL = maskCanvas.toDataURL('image/png');

    fetch('/save_mask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            mask: dataURL,
            filename: currentFile
        })
    })
        .then(res => res.json())
        .then(data => {
            const link = document.createElement('a');
            link.href = data.mask_url;
            link.download = data.filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            showToast("Mask downloaded!");
        });
});

function showToast(msg) {
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 3000);
}
