from flask import Flask
from flask import render_template

from core import DogBreedDetector, DogDetector, HumanDetector, WikiClient, DogBreedPredictor, DogBreedResultsBuilder
from core.models import path_to_tensor, prepare_image, prepare_open_cv_image, dog_breed_predictions
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
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            data["predictions"] = predictor.predict(image)
            data["success"] = data["predictions"] is not None

            if data["predictions"]:
                result_builder = DogBreedResultsBuilder(data['predictions'], wiki_client)
                data['result'] = result_builder.build()
                
    return render_template('index.html', data=data)


if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."))
    global  dog_model, dog_breed_model, input_model, human_model, wiki_client, predictor
    
    input_model, dog_breed_model = DogBreedDetector().load_models()
    dog_model = DogDetector().load_model()
    human_model = HumanDetector().load_model()
    wiki_client = WikiClient()
    predictor = DogBreedPredictor(dog_model, dog_breed_model, input_model, human_model)


    app.run(host='0.0.0.0', port=3001, debug=True)