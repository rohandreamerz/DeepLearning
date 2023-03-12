from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import gevent
import tensorflow

# Keras
from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
import cv2

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'guitarclassifier.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# Necessary


print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    
    img = cv2.imread(img_path)

    # Preprocessing the image
    
    resize = tensorflow.image.resize(img, (256,256))
        

    preds = model.predict(np.expand_dims(resize/255, 0))
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        
          
        if preds > 0.5: 
            result ='Electric Guitar'
        else:
            result = 'Acoustic Guitar'
            
        # Convert to string
        
        return result
    
    return result


if __name__ == '__main__':
    app.run(debug=True)

