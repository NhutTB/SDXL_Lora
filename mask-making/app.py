from flask import Flask, render_template, request, send_file, jsonify
import os
import base64
from PIL import Image
import io
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    filename = str(uuid.uuid4()) + "_" + file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    # Get image dimensions
    with Image.open(filepath) as img:
        width, height = img.size
    
    return jsonify({
        'filename': filename,
        'width': width,
        'height': height,
        'url': f'/uploads/{filename}'
    })

@app.route('/save_mask', methods=['POST'])
def save_mask():
    data = request.json
    mask_data = data['mask'].split(',')[1] # Remove header
    original_filename = data['filename']
    
    # Decode base64
    mask_bytes = base64.b64decode(mask_data)
    mask_img = Image.open(io.BytesIO(mask_bytes))
    
    # Load original to get exact size
    original_path = os.path.join(UPLOAD_FOLDER, original_filename)
    with Image.open(original_path) as org:
        target_size = org.size
    
    # Convert mask to Grayscale ('L' mode) and ensure size
    # The mask_img from frontend is usually RGBA. We need to extract the "drawing" part.
    # Note: The user wants "gray color", so we'll ensure it is saved as L
    mask_l = mask_img.convert('L')
    
    # Save the mask
    mask_filename = "mask_" + original_filename.replace(".jpg", ".png").replace(".jpeg", ".png")
    mask_path = os.path.join(UPLOAD_FOLDER, mask_filename)
    mask_l.save(mask_path)
    
    return jsonify({
        'mask_url': f'/download/{mask_filename}',
        'filename': mask_filename
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    print("Mask Maker server starting...")
    print("Navigate to http://127.0.0.1:5000 in your browser.")
    app.run(debug=True, port=5000)
