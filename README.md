# Food vs. Non-Food Image Classifier
## Food:
![Food](https://raw.githubusercontent.com/elluck91/FoodClassifier/master/img/food.jpg)
## Non-Food:
![Non-Food](https://raw.githubusercontent.com/elluck91/FoodClassifier/master/img/non_food.jpg)
## Notebook description:
The following Notebook was created in order to classify images into two categories: Food, and Non-Food. It relies on Keras, a high-level neural networks API.

### Approach:
The following implementation compares the performance of a Convolutional Neural Network with the following parameters:

-  Number of layers: 1, 2, 3
-  Learning rate: 1e<sup>-3</sup>
-  Batch Size: 32, 100
-  Epochs: 20
-  Image Size: 32x32px, 64x64px
-  Number of kernels in a layer: 10, 20 in the first layer; 50, 100 in the second layer
-  Kernel size: 5x5 small image; 7x7 large image

Each layer follows CONV -> RELU -> MAXPOOL

## Datasets:
The datasets is split three-way:
Training: 60%
Test: 20%
Validation: 20%

Images used in the training/testing/validation are a mix of publicly available datasets.

## Data Preprocessing:
The dataset has been preprocessed by resizing the images into 32x32 and 64x64px. The images remain as RGB, not gray-scale. The argument for keeping the representation as RGB is supported by Kiyoharu Aizawa, in the paper "Food Detection and Recognition Using Convolutional Neural Network," and can be found __[here](https://www.researchgate.net/publication/266357771_Food_Detection_and_Recognition_Using_Convolutional_Neural_Network)__.

Since the dataset used in training is sizable (~5.2k Food, ~4.5k Non-Food), there is no need for augumentation. However, on a limited dataset it would be beneficial to apply augumentation with Keras ImageDataGenerator, using the following techniques:
-  rotation
-  width/height shift
-  sheer range
-  zoom range
-  horizontal flip
***
# Results:
### The best model with one layer has the following parameters:
Layer Size: 20 neurons
Neuron Size: 5x5
Batch Size: 32
![Best one layer model](https://raw.githubusercontent.com/elluck91/FoodClassifier/master/img/one_layer.png)

### The best model with two layers has the following parameters:
Layer One Size: N neurons
Layer Two Size: M neurons
Neuron Size: 5x5
Batch Size: 32
