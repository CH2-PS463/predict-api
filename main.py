import os
from google.cloud import storage
import tensorflow as tf
from io import BytesIO
from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from tensorflow.keras.applications.mobilenet import preprocess_input

app = Flask(__name__)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'satriasayur-credentials.json'
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
                'satriasayur-404406')
            filename = request.json['filename']
            img_blob = image_bucket.blob('predict_uploads/' + filename)
            img_path = BytesIO(img_blob.download_as_bytes())
        except Exception:
            respond = jsonify({'message': 'Error loading image file'})
            respond.status_code = 400
            return respond

        img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
        x = tf.keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        images = np.vstack([x])

        # model predict
        pred_animal = model.predict(images)
        # find the max prediction of the image
        maxx = pred_animal.max()

        nama = ['Bekantan', 'Beruang Madu', 'Burung Rangkong Badak',
                'Gajah Asia', 'Macan Tutul Jawa', 'Tapir Asia']
        nama_saintifik = ['Nasalis Larvatus', 'Helarctos Malayanu', 'Buceroa Rhinocero',
                          'Elephas maximus', 'Panthera pardus melas', 'Tapirus Indicus']
        lokasi = ['Pulau Kalimantan', 'Pulau Kalimantan dan Sumatra', 'Hutan Sumatra, dan Jawa',
                  'Pulau Kalimantan dan Sumatra', 'Pulau Jawa (Jawa barat)', 'Sumatra']
        populasi = ['Terancam Punah (kritis) (International Union for Conversation of Nature)',
                    'Kelangkaan Vulnerable (rentan) (International Union for Conversation of Nature)',
                    'Kelangkaan Vulnerable (rentan) (International Union for Conversation of Nature)',
                    'Terancam, tersisa 41,410 sampai 52,345 di Dunia (menurut Wikipedia)',
                    'Terancam Punah (kritis) (International Union for Conversation of Nature)',
                    'Berpeluang punah lebih 20 persen dalam kurun waktu 20 tahun (International Union for Conversation of Nature)']
        funfact = ['Hidung panjang pada Bekantan hanya ditemui pada species jantan',
                   'Merupakan jenis paling kecil dari jenis beruang yang ada di dunia',
                   'Memiliki raungan yang bersuara "honk"',
                   'Gajah mengkonsumsi sekitar 150 kg (330 pon) pakan tanaman per hari',
                   'Dibandingkan macan tutul lain, macan tutul jawa merupakan spesies paling kecil',
                   'Tapir Memiliki penglihatan yang buruk, namun memiliki indra penciuman dan pendengaran yang kuat']

        # for respond output from prediction if predict <=0.4
        if maxx <= 0.75:
            respond = jsonify({
                'message': 'Hewan tidak terdeteksi'
            })
            respond.status_code = 400
            return respond

        result = {
            "nama": nama[np.argmax(pred_animal)],
            "nama_saintifik": nama_saintifik[np.argmax(pred_animal)],
            "lokasi": lokasi[np.argmax(pred_animal)],
            "populasi": populasi[np.argmax(pred_animal)],
            "funfact": funfact[np.argmax(pred_animal)]
        }

        respond = jsonify(result)
        respond.status_code = 200
        return respond

    return 'OK banh indra'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')