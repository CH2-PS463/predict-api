import os
from google.cloud import storage
import tensorflow as tf
from io import BytesIO
from flask import Flask, request, jsonify
import streamlit as st
import numpy as np

#connection to gcp
app = Flask(__name__)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'capstone-satriasayur.json'
storage_client = storage.Client()

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        #cek if bucket is ready
        try:
            image_bucket = storage_client.get_bucket(
                'satriasayur-bucket')
            filename = request.json['filename']
            img_blob = image_bucket.blob('predict-uploads/' + filename)
            img_path = BytesIO(img_blob.download_as_bytes())
        except Exception:
            respond = jsonify({'message': 'Error loading image file'})
            respond.status_code = 400
            return respond
        
        #Reading Labels
        result_index = model_prediction(img_path)
        with open("labels.txt") as f:
            content = f.readlines()
        label = []
        for i in content:
            label.append(i[:-1])
        st.success("Model is Predicting it's a {}".format(label[result_index]))

        #output model
        result = {
            "nama": format(label[result_index]),
        }

        respond = jsonify(result)
        respond.status_code = 200
        return respond

    return 'OK'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')