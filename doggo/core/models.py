from keras.applications.xception import Xception, preprocess_input
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import ResNet50
from keras.applications import imagenet_utils
from dataclasses import dataclass
from typing import List

from core import dog_names
import numpy as np
import cv2


# MODELS
class DogBreedDetector:
    """ CNN for detecting dog Breeds on photos """

    def load_models(self):
        """
        Load required models:
            - Xception default: for processing input
            - CNN Dog Breed detector: used in the end predictions
        INPUT
        OUTPUT
            Xception_model : Xception default, Keras model
            model: CNN model, Keras model
        """
        Xception_model = Xception(weights='imagenet', include_top=False)
        model = load_model('core/models/dog_breed_detector.h5')

        return Xception_model, model


class DogDetector:
    """ Imagenet model for detecting dog on photos """

    def load_model(self):
        """
        Load ResNet50 model
        INPUT
        OUTPUT
            ResNet50 model : loaded model, Keras model
        """
        return ResNet50(weights='imagenet')


class HumanDetector:
    """ Opencv model for detecting human faces on photos """

    def load_model(self):
        """
        Load Opencv model
        INPUT
        OUTPUT
            Opencv model : loaded model, Opencv model
        """
        return cv2.CascadeClassifier('core/models/haarcascade_frontalface_alt.xml')


# HELPER FUNCTIONS
def path_to_tensor(image):
    """
        Encode PIL.Image for CNN model (Xception)
        INPUT
            image: image uploaded by the user, PIL.image
        OUTPUT
            tensor : image as tensor, numpy object
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    # loads RGB image as PIL.Image.Image type
    img = image.resize((224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def prepare_image(image):
    """
    Encode PIL.Image for Resnet50 model
    INPUT
        image: image uploaded by the user, PIL.image
    OUTPUT
        tensor : image as tensor, numpy object
    """
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image


def _format_breed_name(name):
    """
    Format breed name for displaying
    INPUT
        name: raw breed name, str
    OUTPUT
        name : cleaned breed name, str
    """
    return name.split('.')[1].replace('_', ' ')


def _format_probability(prob):
    """
    Format prediction probability for displaying
    INPUT
        prob: raw model probability, float
    OUTPUT
        label : cleaned probability, str
    """
    return f'{round(100 * prob, 2)}%'


def dog_breed_predictions(predicted_vector, n=3):
    """
    Get dog breed predictions (class + probabilities)
    INPUT
        predicted_vector: vector with each breed probability (last layer of CNN), numpy
        n: number of predictions, int
    OUTPUT
        best_predictions : predictions with the highest probabilities, dict
    """
    best_predictions = np.argsort(predicted_vector * -1).flatten()[:n]
    return dict([
        (_format_breed_name(dog_names[idx]),
         np.take(predicted_vector, [idx][0])
         )
        for idx in best_predictions
    ]
    )


def prepare_open_cv_image(image):
    """
    Process PIL.Image for OpenCV model
    INPUT
        image: uploaded image, PIL.Image
    OUTPUT
        cv2_image : preprocessed image for opencv, cv2 image object
    """
    img = image.copy()
    if img.mode != "RGB":
        img = img.convert("RGB")

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)


# PREDICTION
class DogBreedPredictor:
    """ Dog Breed predictor. Orchestrate all models for generating the end predictions """

    def __init__(self, dog_model, dog_breed_model, input_model, human_model):
        self.dog_model = dog_model
        self.dog_breed_model = dog_breed_model
        self.input_model = input_model
        self.human_model = human_model

    def predict(self, image):
        """
        Predict dog breeds
        INPUT
            image: uploaded image, PIL.Image
        OUTPUT
            predictions : dog breed predictions with their probabilities, dict
        """
        if self._dog_detector(image) or self._face_detector(image):
            return self._dog_breed_detector(image)

        return None

    def _dog_detector(self, image):
        """
        Invoke dog detector model
        INPUT
            image: uploaded image, PIL.Image
        OUTPUT
            prediction: binary classification, boolean
        """
        image = prepare_image(image)
        predicted_vector = self.dog_model.predict(image)
        prediction = np.argmax(predicted_vector)

        return ((prediction <= 268) & (prediction >= 151))

    def _dog_breed_detector(self, image):
        """
        Invoke dog breed detector model (CNN)
        INPUT
            image: uploaded image, PIL.Image
        OUTPUT
            prediction probabilites: vector with prediction probabilities, numpy array
        """
        img_input = self.input_model.predict(preprocess_input(path_to_tensor(image)))
        predicted_vector = self.dog_breed_model.predict(img_input)

        return dog_breed_predictions(predicted_vector)

    def _face_detector(self, image):
        """
        Invoke a human detector model
        INPUT
            image: uploaded image, PIL.Image
        OUTPUT
            prediction: binary classification, boolean
        """
        faces = self.human_model.detectMultiScale(prepare_open_cv_image(image))
        return len(faces) > 0


@dataclass
class DogBreed:
    """ Dog Breed class to represent prediction result """

    name: str
    summary: str
    images: List[str]
    image_src: str
    probability: float
    prob_display: str

    def __lt__(self, other):
        return self.probability < other.probability


class DogBreedResultsBuilder:
    """ Build DogBreed objects for all predictions """

    def __init__(self, predictions, wiki_client):
        self.predictions = predictions
        self.wiki_client = wiki_client

    def build(self):
        """
        Build result set
        INPUT
        OUTPUT
            result set: array of dog breed predictions, array
        """
        result = []

        for breed, prob in self.predictions.items():
            data = self.wiki_client.search(breed)
            result.append(
                DogBreed(name=breed,
                         images=data['images'],
                         summary=data['summary'],
                         image_src=data['image_src'],
                         probability=prob,
                         prob_display=_format_probability(prob))
            )

        return sorted(result, reverse=True)