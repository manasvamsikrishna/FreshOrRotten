from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('./model/model.h5')

# Define class names used during training
class_names = [
    'freshapples',
    'freshbanana',
    'freshoranges',
    'rottenapples',
    'rottenbanana',
    'rottenoranges'
]

# Allowed image formats
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file_path):
    img = load_img(file_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Home route (GET)
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route (POST)
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    os.makedirs('uploads', exist_ok=True)
    filepath = os.path.join('uploads', filename)
    file.save(filepath)

    try:
        img = preprocess_image(filepath)
        predictions = model.predict(img)
        predicted_class = int(np.argmax(predictions[0]))
        label = class_names[predicted_class]
        confidence = float(np.max(predictions[0]))

        return jsonify({
            'predicted_label': label,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

# Local run (ignored by Render, safe to keep)
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
