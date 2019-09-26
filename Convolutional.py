from __future__ import absolute_import, division, print_function, unicode_literals



import tensorflow as tf
from tensorflow import keras


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle


#last excercise was to do this, so im not going show you my script
# Opening the files about data
train_images = pickle.load(open("X.pickle", "rb"))
train_labels = pickle.load(open("y.pickle", "rb"))

#We declare the model as a list of layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(70, 70,1)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

# Compiling the model using some basic parameters
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])





#We are training the model with 100 iterations, 
#Train_images is the array were we load all the training images into a tensor, labels is the array of training labels that match 
# with every image
model.fit(train_images, train_labels, epochs=100)

#We save the model
model.save('model.h5')

