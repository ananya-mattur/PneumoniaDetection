import tensorflow as tf
from keras.optimizers import Adam
import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image
import time
import numpy as np
import h5py
from keras.models import load_model
from keras.optimizers import Adam
import urllib.request

# Download the model files
model_urls = [
    'https://storage.googleapis.com/fp_model/P_CNN.h5',
    'https://storage.googleapis.com/fp_model/P_VGG16_Model.h5'
]
model_files = ['P_CNN.h5', 'P_VGG16_Model.h5']
for url, file in zip(model_urls, model_files):
    urllib.request.urlretrieve(url, file)

# Load the models
models = []
for file in model_files:
    with h5py.File(file, 'r') as f:
        model = load_model(f, compile=False)
        models.append(model)

# Define the ensemble function that makes predictions using all the models and takes the average
def ensemble_predict(img):
    predictions = []
    for model in models:
        prediction = model.predict(img)
        predictions.append(prediction)
    predictions = np.array(predictions)
    avg_prediction = np.mean(predictions, axis=0)
    return avg_prediction

app = Flask(__name__)

# Define allowed image file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Define a function to check if an uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define the Flask route for the main page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/howtouse')
def howtouse():
    return render_template('howtouse.html')

# Define the Flask route for handling image uploads and making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']

        # Generate a unique filename
        filename = secure_filename(file.filename)
        filename = str(time.time()) + '_' + filename

        # Save the file in the 'temp' directory
        file_path = os.path.join(os.getcwd(), 'temp', filename)
        file.save(file_path)

        # Load the saved image
        img = Image.open(os.path.join('temp', filename))

        # Preprocess the image
        img = img.resize((224, 224))
        img = img.convert('RGB')
        img = np.asarray(img)
        img = np.expand_dims(img, axis=0)

        # Make a prediction using the ensemble model
        prediction = ensemble_predict(img)
        if prediction[0] < 0.5:
            result = 'NEGATIVE'
        else:
            result = 'POSITIVE'

        # Return the prediction result as a template variable
        return render_template('predict.html', result=result, filename=filename)
    
    # If the request method is not POST, return the index page
    return render_template('index.html')

# Define a Flask route for handling 404 errors
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(debug=True)
  
