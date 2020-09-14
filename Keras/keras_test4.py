# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 17:48:55 2020

@author: lenovo
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.applications.vgg16 import VGG16
import keras

model = VGG16(include_top = False, weights = 'imagenet', input_shape = (150,150,3))
print('load model ok')

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    r'C:\Users\lenovo\Desktop\Keras\train',
    target_size=(150,150),
    batch_size = 20,
    class_mode = None,
    shuffle = False
    )

test_generator = datagen.flow_from_directory(
    r'C:\Users\lenovo\Desktop\Keras\test',
    target_size = (150,150),
    batch_size = 20,
    class_mode = None,
    shuffle = False
    )
print('increase images ok')

WEIGTS_PATH = ''
model.load_weights(r'C:\Users\lenovo\.keras\models\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
print('load pre model ok')

bottleneck_features_train = model.predict_generator(train_generator,20)
np.save(r'C:\Users\lenovo\Desktop\Keras\bottleneck_features_train.npy',bottleneck_features_train)

bottleneck_features_validation = model.predict_generator(test_generator,5)
np.save(r'C:\Users\lenovo\Desktop\Keras\bottleneck_features_validation.npy',bottleneck_features_validation)

train_data = np.load(open(r'C:\Users\lenovo\Desktop\Keras\bottleneck_features_train.npy','rb'))
train_labels = np.array([0] * 80 + [1] * 80 + [2] * 80 + [3] * 80 + [4] * 80)  # matt,打标签

validation_data = np.load(open(r'C:\Users\lenovo\Desktop\Keras\bottleneck_features_validation.npy','rb'))
validation_labels = np.array([0] * 20 + [1] * 20 + [2] * 20 + [3] * 20 + [4] * 20)  # matt,打标签

train_labels = keras.utils.to_categorical(train_labels, 5)
validation_labels = keras.utils.to_categorical(validation_labels,5)

l_model = Sequential()
l_model.add(Flatten(input_shape=(4,4,512)))
l_model.add(Dense(256, activation='relu'))
l_model.add(Dropout(0.5))
l_model.add(Dense(5, activation='softmax'))

l_model.compile(loss = 'categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

l_model.fit(train_data, train_labels,
            nb_epoch = 50, batch_size=16,
            validation_data=(validation_data, validation_labels))
