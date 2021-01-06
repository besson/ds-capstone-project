from flask import Flask

from core import DogBreedDetector, DogDetector, HumanDetector
from core.models import path_to_tensor, dog_breed_prediction, prepare_image, prepare_open_cv_image
from keras.applications.xception import preprocess_input
from PIL import Image

import flask
import numpy as np
import io
import cv2


# Based on https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
app = Flask(__name__)    

@app.route('/')
@app.route('/index')
def index():
    return 'Hello doggo'

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            data["predictions"] = _predict(image)
            data["success"] = True

    return flask.jsonify(data)


def _predict(image):
    if _dog_detector(image) or _face_detector(image):
        return _dog_breed_detector(image)

    return 'not dog or human face'


def _dog_detector(image):
    image = prepare_image(image)
    predicted_vector = dog_model.predict(image)
    prediction = np.argmax(predicted_vector)

    return ((prediction <= 268) & (prediction >= 151))
    
    
def _dog_breed_detector(image):
    img_input = input_model.predict(preprocess_input(path_to_tensor(image)))
    predicted_vector = dog_breed_model.predict(img_input)

    return dog_breed_prediction(predicted_vector)


def _face_detector(image):
    faces = human_model.detectMultiScale(prepare_open_cv_image(image))
    return len(faces) > 0


if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."))
    global dog_model
    global dog_breed_model
    global input_model
    global human_model

    input_model, dog_breed_model = DogBreedDetector().load_models()
    dog_model = DogDetector().load_model()
    human_model = HumanDetector().load_model()

    app.run(host='0.0.0.0', port=3001, debug=True)