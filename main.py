import os
from google.cloud import storage
import tensorflow as tf
from io import BytesIO
from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from tensorflow.keras.applications.mobilenet import preprocess_input

app = Flask(__name__)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'capstone-satriasayur.json'
storage_client = storage.Client()


def req(y_true, y_pred):
    req = tf.metrics.req(y_true, y_pred)[1]
    tf.keras.backend.get_session().run(tf.local_variables_initializer())
    return req


# model_filename = 'my_model_fix.h5'
# model_bucket = storage_client.get_bucket('sa-lindungi-model-bucket')
# model_blob = model_bucket.blob(model_filename)
# model_blob.download_to_filename(model_filename)
# model = load_model(model_filename, custom_objects={'req': req})
model = load_model('trained_model.h5', custom_objects={'req': req})


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
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

        img = tf.keras.utils.load_img(img_path, target_size=(64, 64))
        x = tf.keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        images = np.vstack([x])

        # model predict
        pred_vegetables = model.predict(images)
        # find the max prediction of the image
        maxx = pred_vegetables.max()

        nama = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd',
                'Brinjal', 'Broccoli', 'Cabbage','Capsicum',
                'Carrot','Cauliflower','Cucumber','Papaya',
                'Potato','Pumpkin','Radish','Tomato']

        # for respond output from prediction if predict <=0.4
        if maxx <= 0.75:
            respond = jsonify({
                'message': 'Sayuran tidak terdeteksi'
            })
            respond.status_code = 400
            return respond

        result = {
            "nama": nama[np.argmax(pred_vegetables)],
            "value_var": print(pred_vegetables)
        }

        respond = jsonify(result)
        respond.status_code = 200
        return respond

    return 'OK'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')