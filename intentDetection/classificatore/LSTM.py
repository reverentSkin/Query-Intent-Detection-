#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 00:47:35 2019

@author: mazzone
"""
# Keras

from keras.models import Sequential
from keras.layers import Dense,LSTM, Dropout, Activation

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np
import joblib


def create_model():
    mod = Sequential()
    mod.add(LSTM(300, activation='relu'))
    mod.add(Dense(200, activation='relu'))
    mod.add(Dense(4, activation='softmax'))
    mod.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return mod

#class_weight = {0: 1.8,
 #               1: 3.2,
  #              2: 1.,
   #             3: 1.2}

#training3D_matrix = joblib.load('../dataPreProcessing/matrix3D_TRAININGANSWER_word2vecNOLEMMA')
#testing3D_matrix = joblib.load('../dataPreProcessing/matrix3D_TESTINGANSWER_word2vecNOLEMMA')
training3D_matrix = joblib.load('../dataPreProcessing/matrix3D_TRAININGANSWER_fastTextNOLEMMA')
testing3D_matrix = joblib.load('../dataPreProcessing/matrix3D_TESTINGANSWER_fastTextNOLEMMA')

trainingSet = pd.read_csv('../dataPreProcessing/trainingSet1165.csv', sep = ',')
testingSet = pd.read_csv('../dataPreProcessing/testingSet.csv', sep = ';')
cat1=np.array(trainingSet['categoria'])
cat2=np.array(testingSet['categoria'])
label_encoder = LabelEncoder()
trainY = label_encoder.fit_transform(cat1)
testY= label_encoder.fit_transform(cat2)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
trainY = trainY.reshape(len(trainY), 1)
trainY = onehot_encoder.fit_transform(trainY)

testY = testY.reshape(len(testY), 1)
testY = onehot_encoder.fit_transform(testY)

model = create_model()

model.fit(training3D_matrix, trainY, validation_data=(testing3D_matrix, testY), epochs=20, batch_size=32, class_weight='auto')
#model.fit(training3D_matrix, trainY, validation_data=(testing3D_matrix, testY), epochs=20, class_weight='auto')

#joblib.dump(model, 'word2vecNOlemLSTMv6')
joblib.dump(model, 'fastTextNOlemLSTMv7')