#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 11:39:33 2019

@author: mazzone
"""
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Bidirectional, GlobalMaxPool1D
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold



def create_model():
    mod = Sequential()
    mod.add(Dropout(0.2))
    mod.add(Conv1D(100, 4, activation='relu', input_shape=(50,300)))
    mod.add(MaxPooling1D(pool_size=4))
    mod.add(Bidirectional(LSTM(200, activation ='relu', return_sequences = True)))
    mod.add(GlobalMaxPool1D())
    mod.add(Dense(200, activation='relu'))
    mod.add(Dropout(0.2))
    mod.add(Dense(4, activation='softmax'))
    mod.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return mod

def train_model(Xtrain, Ytrain):
    model = create_model()
    model.fit(training3D_matrix, trainY, epochs=20, class_weight='auto')
    return model

def test_model(model, Xtest, YTest):
    
    test_accuracy = model.evaluate(Xtest, YTest)
    return test_accuracy



training3D_matrix = joblib.load('../dataPreProcessing/matrix3D_TRAININGANSWER_word2vecNOLEMMA')
testing3D_matrix = joblib.load('../dataPreProcessing/matrix3D_TESTINGANSWER_word2vecNOLEMMA')
#training3D_matrix = joblib.load('../dataPreProcessing/matrix3D_TRAININGANSWER_fastTextNOLEMMA')
#testing3D_matrix = joblib.load('../dataPreProcessing/matrix3D_TESTINGANSWER_fastTextNOLEMMA')
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

# prepare the k-fold cross-validation configuration
n_folds = 10
kfold = KFold(n_folds, True, 1)
X=training3D_matrix#np.arange(training3D_matrix,testing3D_matrix)

y=trainY#np.arange(trainY,testY)
print(y)
# cross validation estimation of performance
scores, members = list(), list()
for train_ix, test_ix in kfold.split(X):
        trainX, trainy = X[train_ix], y[train_ix]
        testX, testy = X[test_ix], y[test_ix]
        print(trainX)
        mode = train_model(trainX, trainy)
        test_acc = test_model(mode,testX, testy)
        print(test_acc)
        scores.append(test_acc)
        members.append(mode)
print(scores)
    