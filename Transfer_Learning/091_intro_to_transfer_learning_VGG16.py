#!/usr/bin/env python
__author__ = "Omid Charrakh"
__license__ = "This is based on Python for Microscopists"

# https://youtu.be/LOvrfvtiC8c

#Load the VGG model. For the first time, it downloads weights from the Internet.
#Stored in Keras/Models directory. (Almost 600MB)
#We can include arguments to define whether we want to download full model, or parts only.

from keras.applications.vgg16 import VGG16
# Load the model
model = VGG16()
model.summary()


img_dir='/Users/omid/Documents/GitHub/statistics/Image-Processing/My_Practice/Keras_PlayList/images/'

#Let us load an image to test the pretrained VGG model.
#For VGG16 the images need to be 224x224. 
from keras.preprocessing.image import load_img
image = load_img(img_dir+'/cab.jpg', target_size=(224, 224))

#Convert pixels to Numpy array                                        
from keras.preprocessing.image import img_to_array
image = img_to_array(image)

# VGG expects multiple images of size 224x224x3; so the input needs to be (1, 224, 224, 3)

#Method1:   
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

#Method2:
'''
import numpy as np
image = np.expand_dims(image, axis=0)
'''


#To get best results, data should be preprocessed in the same way as the training dataset: 
# preprocessing_input from Keras does this job. 
# Notice the change in pixel values (Preprocessing_input subtracts mean RGB value of training set from each pixel)

from keras.applications.vgg16 import preprocess_input
image = preprocess_input(image)


# Predict the probability across all output categories (for each of the 1000 classes)

prediction = model.predict(image)

#Print the probabilities of the top 5 classes
from tensorflow.keras.applications.mobilenet import decode_predictions
pred_classes = decode_predictions(prediction, top=5)
for i in pred_classes[0]:
    print(i)