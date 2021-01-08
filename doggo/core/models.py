import keras

from keras.applications.xception import Xception, preprocess_input
from keras.models import load_model
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications import imagenet_utils
from dataclasses import dataclass
from typing import List

from core import dog_names
import numpy as np
import cv2                


class DogBreedDetector:

    def load_models(self):
        Xception_model = Xception(weights='imagenet', include_top=False)
        model = load_model('core/models/dog_breed_detector.h5')

        return Xception_model, model

def path_to_tensor(image):
    if image.mode != "RGB":
        image = image.convert("RGB")

    # loads RGB image as PIL.Image.Image type
    img = image.resize((224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

class DogDetector:

    def load_model(self):
        return ResNet50(weights='imagenet')

class HumanDetector:

    def load_model(self):
        return cv2.CascadeClassifier('core/models/haarcascade_frontalface_alt.xml')


class DogBreedPredictor:

    def __init__(self, dog_model, dog_breed_model, input_model, human_model):
        self.dog_model = dog_model
        self.dog_breed_model = dog_breed_model
        self.input_model = input_model
        self.human_model = human_model

    def predict(self, image):
        if self._dog_detector(image) or self._face_detector(image):
            return self._dog_breed_detector(image)

        return None
    
    def _dog_detector(self, image):
        image = prepare_image(image)
        predicted_vector = self.dog_model.predict(image)
        prediction = np.argmax(predicted_vector)

        return ((prediction <= 268) & (prediction >= 151))

    def _dog_breed_detector(self, image):
        img_input = self.input_model.predict(preprocess_input(path_to_tensor(image)))
        predicted_vector = self.dog_breed_model.predict(img_input)

        return dog_breed_predictions(predicted_vector)

    def _face_detector(self, image):
        faces = self.human_model.detectMultiScale(prepare_open_cv_image(image))
        return len(faces) > 0

def prepare_image(image):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image

def _format_breed_name(name: str):
    return name.split('.')[1].replace('_', ' ')

def _format_probability(prob: float):
    return f'{round(100 * prob, 2)}%' 

def dog_breed_predictions(predicted_vector, n=3):
    best_predictions = np.argsort(predicted_vector * -1).flatten()[:n]
    return dict([
                (_format_breed_name(dog_names[idx]),
                _format_probability(np.take(predicted_vector, [idx][0]))
                )
                for idx in best_predictions
                ]
            )
    

def prepare_open_cv_image(image):
    img = image.copy()
    if img.mode != "RGB":
        img = img.convert("RGB")

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    
@dataclass
class DogBreed:

    name: str
    summary: str
    images: List[str]
    probability: float

    def __lt__(self, other):
        return self.probability < other.probability


class DogBreedResultsBuilder:

    def __init__(self, predictions, wiki_client):
        self.predictions = predictions
        self.wiki_client = wiki_client
        
    def build(self):
        result = []

        for breed, prob in self.predictions.items():
            data = self.wiki_client.search(breed)
            result.append(
                        DogBreed(name=breed,
                                images=data['images'],
                                summary=data['summary'],
                                probability=prob)
            )
        
        return sorted(result, reverse=True)