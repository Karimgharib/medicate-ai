import tensorflow as tf
import cv2
import numpy as np
from flask import Flask, request, jsonify
from keras.models import load_model
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

skinModel = load_model('skin_model.h5')
chestModel = load_model('chest_model.h5')
brainModel = load_model('brain.hdf5')





@app.route('/api/predict/chest', methods=['POST'])
def predictChest():
    

    file = request.files['file']
    img_bytes = file.read()

    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    img_resized = cv2.resize(img, (224, 224))

    img_array = np.array(img_resized) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    pred = chestModel.predict(img_array)

    output_class = np.argmax(pred)

    if output_class == 0:
        return jsonify({'result': 'covid'})
    elif output_class == 1:
        return jsonify({'result': 'normal'})
    else:
        return jsonify({'result': 'unknown'})


@app.route('/api/predict/skin', methods=['POST'])
def predictSkin():
    

    # Get the image from the request
    file = request.files.get('file')
    img_bytes = file.read()

    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    img_resized = cv2.resize(img, (224, 224))

    img_array = np.array(img_resized) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction

    probabilities = skinModel.predict(img_array)

    class_idx = np.argmax(probabilities)

    # output_class = {classes[class_idx]: probabilities[class_idx]}

    if class_idx == 0:
        return jsonify({'result': 'melanoma'})
    elif class_idx == 1:
        return jsonify({'result': 'nevus'})
    elif class_idx == 2:
        return jsonify({'result': 'seborrheic_keratosis'})
    else:
        return jsonify({'result': 'unknown'})
    


@app.route('/api/predict/brain', methods=['POST'])
def predictBrain():
    

    # Get the image from the request
    file = request.files.get('file')
    img_bytes = file.read()

    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    img_resized = cv2.resize(img, (224, 224))

    img_array = np.array(img_resized) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction

    result = brainModel.predict(img_array)
    predictions = [1 if x > 0.5 else 0 for x in result]

    if predictions[0] == 0:
        return jsonify({'result': 'no_tumor'})
    elif predictions[0] == 1:
        return jsonify({'result': 'tumor'})
    else:
        return jsonify({'result': 'unknown'})


if __name__ == '__main__':
    app.run()
