import keras

from keras.applications.xception import Xception, preprocess_input
from keras.models import load_model
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications import imagenet_utils


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

def dog_breed_prediction(predicted_vector):
    return dog_names[np.argmax(predicted_vector)].split('/')[1]
    
def prepare_open_cv_image(image):
    img = image.copy()
    if img.mode != "RGB":
        img = img.convert("RGB")

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    

