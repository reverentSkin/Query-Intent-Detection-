#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 19:27:43 2019

@author: mazzone
"""
# Keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Bidirectional, GlobalMaxPool1D

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import joblib


def create_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(200, activation ='relu', return_sequences = True)))
    model.add(GlobalMaxPool1D())
    #model.add(Dense(300, activation='relu'))
    #model.add(Dropout(0.3))
    model.add(Dense(4, activation='softmax'))
    model.compile ( loss='categorical_crossentropy' , optimizer='adam' , metrics=['accuracy'] )
    return model

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

#model.fit(training3D_matrix, trainY, validation_data=(testing3D_matrix, testY), epochs=20, batch_size=32, class_weight='auto')
model.fit(training3D_matrix, trainY, validation_data=(testing3D_matrix, testY), epochs=20, class_weight='auto')

#joblib.dump(model, 'word2vecNOlemBiLSTMv7')
joblib.dump(model, 'fastTextNOlemBiLSTMv7')
