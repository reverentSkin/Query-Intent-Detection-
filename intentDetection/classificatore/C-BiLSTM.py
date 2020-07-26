#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 09:34:57 2019

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

from sklearn.manifold import TSNE

def create_model():
    mod = Sequential()
    mod.add(Dropout(0.2))
    mod.add(Conv1D(64, 4, activation='relu', input_shape=(50,300)))
    mod.add(MaxPooling1D(pool_size=4))
    mod.add(Bidirectional(LSTM(200, activation ='relu', return_sequences = True)))
    mod.add(GlobalMaxPool1D())
    mod.add(Dense(300, activation='relu'))
    mod.add(Dropout(0.3))
    mod.add(Dense(4, activation='softmax'))
    mod.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return mod


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

#model.fit(training3D_matrix, trainY, validation_data=(testing3D_matrix, testY), epochs=40, batch_size=32, class_weight='auto')
model.fit(training3D_matrix, trainY, validation_data=(testing3D_matrix, testY), epochs=40, class_weight='auto')

#joblib.dump(model, 'word2vecNOlemC-BiLSTMv6')
joblib.dump(model, 'fastTextNOlemC-BiLSTMv7')