# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 19:54:53 2021

@author: ted-17
"""

import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from models import make_model
from utils import make_datasets
from glob import glob
from const import N_CATEGORIES


if __name__ == '__main__':
    #-- prepare wavpaths list (dummy data)
    wavpaths=glob('*.wav')
    labels=np.ones(len(wavpaths))
    
    #-- extract feature
    X=make_datasets(wavpaths)
    
    #-- make model
    model=make_model(N_CATEGORIES)
    
    #-- train test split
    Y=np_utils.to_categorical(labels, len(labels))
    X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=.2,random_state=42)
    
    #-- model training
    result=model.fit(X_train, y_train, batch_size=32, epochs=40)
    
    #-- model evaluation
    model.evaluate(X_test, y_test)