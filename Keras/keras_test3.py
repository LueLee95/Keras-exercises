# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 20:10:19 2020

@author: lenovo
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
import matplotlib.pyplot as plt

def built_model():
    model =  Sequential()
    model.add(Conv2D(32,(3,3), input_shape=(150,150,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    
    model.compile(loss = 'categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    model.summary()
    
    return model
    
def generate_data():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        r'C:\Users\lenovo\Desktop\Keras\train',
        target_size=(150,150),
        batch_size = 32,
        class_mode = 'categorical'
        #save_to_dir=r'C:\Users\lenovo\Desktop\Keras\gen'
        )
    
    validation_generator = test_datagen.flow_from_directory(
        r'C:\Users\lenovo\Desktop\Keras\test',
        target_size=(150,150),
        batch_size = 32,
        class_mode = 'categorical'
        )
    
    return train_generator, validation_generator

def train_model (model = None):
    if model is None:
        model = built_model()
        model.fit_generator(
            train_generator,
            samples_per_epoch = 2000, 
            nb_epoch = 5,
            validation_data=validation_generator,
            nb_val_samples = 800
            )
        model.save_weights('first_try_animal.h5')
    
    return model

def plot_training(history):
    plt.figure(12)
    plt.subplot(121)
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(train_acc))
    plt.plot(epochs, train_acc, 'b', label = 'train_acc')
    plt.plot(epochs, val_acc, 'r', label = 'test_acc')
    plt.title('Train and Test accuracy')
    plt.legend()
    
    plt.subplot(122)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(train_loss))
    plt.plot(epochs, train_loss, 'b', label = 'train_loss')
    plt.plot(epochs, val_loss, 'r', label = 'test_loss')
    plt.title('Train and Test loss')
    plt.legend()
    
    plt.show()
    

    
if __name__ == '__main__':
    train_generator, validation_generator = generate_data()
    model=train_model()

    