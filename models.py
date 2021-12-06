# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 19:52:16 2021

@author: ted-17
"""
# using keras VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD,Adam
from keras.callbacks import CSVLogger

def make_model(n_categories):
    base_model=VGG16(weights='imagenet',include_top=False,
                     input_tensor=Input(shape=(224,224,3)))
    
    #add new layers instead of FC networks
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x)
    prediction=Dense(n_categories,activation='softmax')(x)
    model=Model(inputs=base_model.input,outputs=prediction)
    
    #fix weights before VGG16 14layers
    for layer in base_model.layers[:15]:
        layer.trainable=False
    
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model