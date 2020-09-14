# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:04:34 2020

@author: lenovo
"""

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

model=VGG16(weights='imagenet')

img_path=r'C:\Users\lenovo\Desktop\LEE\dog.jpg'
img = image.load_img(img_path,target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
x = preprocess_input(x)

features=model.predict(x)
print('Predicted:', decode_predictions(features, top=3)[0])
result=decode_predictions(features, top=3)[0]
