from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os

app = Flask(__name__)

# Load the Lung Cancer Detection model
lung_model_path = "/Users/akash/MCA Academics/Minor Project/Cancer Detection Project/model.h5"
lung_model = load_model(lung_model_path)

# Load the Breast Cancer Detection model
breast_model_path = "/Users/akash/MCA Academics/Minor Project/Cancer Detection Project/breast_cancer_model.h5"
breast_model = load_model(breast_model_path)

# Define function to preprocess image for Lung Cancer Detection
def preprocess_lung_image(file_path):
    img = image.load_img(file_path, target_size=(64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

# Define function to preprocess image for Breast Cancer Detection
def preprocess_breast_image(file_path):
    img = image.load_img(file_path, target_size=(48, 48))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize the image
    return img

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the home page
@app.route('/home')
def home():
    return render_template('home.html')

# Route for the lung cancer detection page
@app.route('/lung-detection')
def lung_detection():
    return render_template('lung_detection.html')

# Route for the breast cancer detection page
@app.route('/breast-detection')
def breast_detection():
    return render_template('breast_detection.html')

# Route to handle image upload and prediction for both Lung and Breast Cancer Detection
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Save the uploaded file temporarily
        file_path = 'temp_img.jpg'
        file.save(file_path)

        # Preprocess the image for Lung Cancer Detection
        processed_lung_img = preprocess_lung_image(file_path)
        lung_result = lung_model.predict(processed_lung_img)
        lung_prediction = 'Normal' if lung_result[0][0] == 1 else 'Cancer'

        # Preprocess the image for Breast Cancer Detection
        processed_breast_img = preprocess_breast_image(file_path)
        breast_result = breast_model.predict(processed_breast_img)
        breast_prediction = 'Benign' if breast_result[0][0] > 0.5 else 'Malignant'

        # Remove the temporary file
        os.remove(file_path)

        return jsonify({'lung_prediction': lung_prediction, 'breast_prediction': breast_prediction})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
