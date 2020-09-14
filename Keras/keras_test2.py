# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 10:38:12 2020

@author: lenovo
"""

import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img

datagen=ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

dog_path=r'C:\Users\lenovo\Desktop\LEE\dog.jpg'
img = load_img(dog_path)
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

os.mkdir(r'C:\Users\lenovo\Desktop\LEE\dogs')

i = 0
for batch in datagen.flow(x, batch_size=1, 
                          save_to_dir=r'C:\Users\lenovo\Desktop\LEE\dogs',
                          save_prefix='datagen',
                          save_format='jpg'):
    i +=1 
    if i>20:
        break
    
'''
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    r'train/dog',
    target_size=(150,150),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    'data/validation',
    target_size=(150,150),
    batch_size=32,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs = 50,
    validation_data = validation_generator,
    validation_steps = 800)
'''
