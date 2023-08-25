## **Hand-Gesture-Recognition-Using-CNN**
This project is about a simple application of Convolution Neural Networks to detect different hand gestures. The Cascade classifier function extracts the hand image from webcam and uses it to train as well predict the type of gesture that is. More information about the algorithm can be found below.

## **Libraries Requirement**

- Numpy
- Tensorflow
- Scikit-Learn
- Matplotlib
- Seaborn
- OpenCV
- Keras

## **File Description**

- Gesture.py : This file is the code for import the image dataset, split them into train and test data , preprocessing the images and train the model using CNN. To use the code, change the directory name to your preference.
- camera.py : This file is used to get real life input from the device camera through a localhost network.
- model.py : This file is udsed to collect the data from camera.py and predict the data with respect to the classes with our trained model

## **Architecture Insights**

### ImageDataGenerator: 
ImageDataGenerator is used to take the inputs of the original data and then transform it on a random basis, returning the output resultant containing solely the newly changed data. We used this function to reshape, resize, images to grayscale, etc.

### Convulational Neural Network : 
A Convolutional Neural Network (CNN) is a Deep Learning algorithm that can take in an input image, assign importance to various aspects in the image, and be able to differentiate one from the other. We used 4 CNN layers with softmax activation state and 2 fully connected CNN layers.The network is trained with batch size of 128 and epochs of 10.

### Matplotlib and Seaborn : 
These libraries were used to visualize the confusion matrix and elaborate the view of accuracy and precisions.

### OpenCV:
OpenCV module is used to get input from device camera and extract the hand image from the camera. 
