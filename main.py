from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
from PIL import UnidentifiedImageError
import numpy as np
import os
import uuid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.h5')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model(MODEL_PATH)

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Helper function to predict tumor type
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score


def allowed_file(filename):
    _, extension = os.path.splitext(filename.lower())
    return extension in ALLOWED_EXTENSIONS


def build_upload_name(filename):
    safe_name = secure_filename(filename)
    _, extension = os.path.splitext(safe_name)
    return f"{uuid.uuid4().hex}{extension.lower()}"

# Route for the main page (index.html)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        file = request.files.get('file')
        if file is None:
            return render_template('index.html', result=None, error='Please choose an MRI image before submitting.')

        if not file.filename:
            return render_template('index.html', result=None, error='Please choose an MRI image before submitting.')

        if not allowed_file(file.filename):
            return render_template('index.html', result=None, error='Unsupported file type. Upload a PNG, JPG, JPEG, BMP, GIF, or WEBP image.')

        saved_filename = build_upload_name(file.filename)
        file_location = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)

        try:
            file.save(file_location)
            result, confidence = predict_tumor(file_location)
        except (UnidentifiedImageError, OSError, ValueError):
            if os.path.exists(file_location):
                os.remove(file_location)
            return render_template('index.html', result=None, error='The uploaded file could not be processed as a valid MRI image.')

        return render_template(
            'index.html',
            result=result,
            confidence=f"{confidence * 100:.2f}%",
            file_path=f'/uploads/{saved_filename}',
            error=None,
        )

    return render_template('index.html', result=None, error=None)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
