# Udacity's Nanodegree in Data Science: Capstone Project

The Capstone project is divided into 2 main sub-projects:

## Dog project 
In this project, I've implemented a [Convolutional Neural Network (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network) using [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) to detect dog breeds from a photo.
The pre-computed network used is [Xception](https://arxiv.org/abs/1610.02357). Udacity instructors provided guidelines, notebooks, code templates and base models for this exercise. 
Please follow [dog-project/README.md](https://github.com/besson/ds-capstone-project/blob/master/dog-project/README.md) for more information.

All code to train, valid and test different networks, saving the model as well as the analysis to evaluate the best network can be found at [dog-project/dog_app.ipynb](https://github.com/besson/ds-capstone-project/blob/master/dog-project/dog_app.ipynb).
This notebook template was provided by Udacity and extended by me during the project.


## Doggo
Doggo is a Flask web app application to implement the dog-project's models. Based on a photo upload by a user, Doggo runs the following steps:

1. Encode user photo for Dog detector, human face detector and Xception model
2. If photo is classified as a dog or a human face, it predicts dog breeds
3. For the top n predictions, Doggo calls [Wikipedia](http://wikipedia.com) and [dog.ceo](https://dog.ceo) API to find more information about the predicted breeds
4. Return results to user

All code can be found at [doggo folder](https://github.com/besson/ds-capstone-project/tree/master/doggo) and more information about the project at [doggo/README.md](https://github.com/besson/ds-capstone-project/tree/master/doggo/README.md).



