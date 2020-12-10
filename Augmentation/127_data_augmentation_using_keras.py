#!/usr/bin/env python
__author__ = "Omid Charrakh"
__license__ = "This is based on Python for Microscopists"


# https://youtu.be/ccdssX4rIh8
"""
@author: Omid Charrakh

Image shifts via the width_shift_range and height_shift_range arguments.
Image flips via the horizontal_flip and vertical_flip arguments.
Image rotations via the rotation_range argument
Image brightness via the brightness_range argument.
Image zoom via the zoom_range argument.
"""

######################################################################
import tensorflow as tf
import keras 
tf.__version__
keras.__version__


from keras.preprocessing.image import ImageDataGenerator
#ImageDataGenerator is for data augmentation
from skimage import io 
#io is for reading single images

######################################################################

# Construct an instance of the ImageDataGenerator class
# Pass the augmentation parameters through the constructor. 

datagen = ImageDataGenerator(
        rotation_range=45,     #Random rotation between 0 and 45
        width_shift_range=0.2,   #% shift
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2, 
        horizontal_flip=True, # vertical_flip is also possible
        fill_mode='constant', cval=125)  #How to fill the remaining parts of the image: | reflect mode is better for GAN | Also try nearest, constant, reflect, wrap | cval={0=black, 125=gray, 255=white} |


######################################################################
# Loading a single image and using .flow method to augment the image

# To Loading a sample image, you can use any library (like opencv). But they need to be in an array form. 
# If using keras, load_img convert it to an array first.

x = io.imread('/Users/omid/Documents/GitHub/statistics/Image-Processing/My_Practice/Keras_PlayList/images/einstein_mona_lisa/monalisa_original.jpg')  #Array with shape (256, 256, 3)

# Reshape the input image because ...
#x: Input data to datagen.flow must be Numpy array of rank 4 or a tuple.
#First element represents the number of images, second and third are x_pixel and y_pixel and forth is channels numbers (0 if graysclae and 3 if colorful) 

x = x.reshape((1, ) + x.shape)  #Array with shape (1, 256, 256, 3)


# Create 20 augmented images from the monalisa_original.jpg
i = 0
for batch in datagen.flow(x, batch_size=16,  
                          save_to_dir='/Users/omid/Documents/GitHub/statistics/Image-Processing/My_Practice/Keras_PlayList/images/einstein_mona_lisa/monalisa_augmented', 
                          save_prefix='aug', 
                          save_format='png'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely  


####################################################################
#Multiple images (Method 1)

#Manually read each image and create an array to be supplied to datagen via flow method

#Create an iterator by using image dataset in memory (using flow() function)
#Create an iterator by using image dataset from a directory (using flow_from_directory)


import numpy as np
import os
from PIL import Image

dataset = []
image_directory = '/Users/omid/Documents/GitHub/statistics/Image-Processing/My_Practice/Keras_PlayList/images/einstein_mona_lisa/multiclass1/'
my_images = os.listdir(image_directory)

for i, image_name in enumerate(my_images):
    if (image_name.split('.')[1] == 'jpg'):
        image = io.imread(image_directory + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((128,128))
        dataset.append(np.array(image))

x_1 = np.array(dataset)



i = 0
for batch in datagen.flow(x_1, batch_size=16,  
                          save_to_dir='/Users/omid/Documents/GitHub/statistics/Image-Processing/My_Practice/Keras_PlayList/images/einstein_mona_lisa/augmented1/', 
                          save_prefix='aug', 
                          save_format='png'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely  



#####################################################################
#Multiclass images (Mthod 2)
# Read dirctly from the folder structure using flow_from_directory

#This method (i. e. from_directory) is useful if subdirectories are organized by class

i = 0
for batch in datagen.flow_from_directory(directory='/Users/omid/Documents/GitHub/statistics/Image-Processing/My_Practice/Keras_PlayList/images/einstein_mona_lisa/multiclass2', 
                                         batch_size=10,  
                                         target_size=(256, 256),
                                         color_mode="rgb",
                                         save_to_dir='/Users/omid/Documents/GitHub/statistics/Image-Processing/My_Practice/Keras_PlayList/images/einstein_mona_lisa/augmented2/', 
                                         save_prefix='aug', 
                                         save_format='png'):
    i += 1
    if i > 20:
        break 

#####################################################################
#Fitting model with augmented data
#Once data is augmented, you can use it to fit a model via: fit.generator(), instead of fit():
#model.fit_generator(datagen.flow(x))