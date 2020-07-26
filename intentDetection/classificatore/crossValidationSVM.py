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
import pandas as pd


from sklearn import svm
from sklearn.metrics import accuracy_score

def encodeCategory(data):
    Y = np.array(data['categoria'])
    Encoder = LabelEncoder()
    Y = Encoder.fit_transform(Y)
    return Y
def train_model(Xtrain, Ytrain):
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto') 
    model = SVM.fit(Xtrain,Ytrain)
    return model

def test_model(model, Xtest, Ytest):
    prediction = model.predict(Xtest)
    test_accuracy = accuracy_score(prediction, Ytest)
    return test_accuracy






trainX = joblib.load('../dataPreProcessing/trainingTFIDF')

#training3D_matrix = joblib.load('../dataPreProcessing/matrix3D_TRAININGANSWER_fastTextNOLEMMA')
#testing3D_matrix = joblib.load('../dataPreProcessing/matrix3D_TESTINGANSWER_fastTextNOLEMMA')
trainingSet = pd.read_csv('../dataPreProcessing/trainingSet1165.csv', sep = ',')

cat1=np.array(trainingSet['categoria'])

label_encoder = LabelEncoder()
trainY = label_encoder.fit_transform(cat1)

# binary encode
trainY=encodeCategory(trainingSet)
print(trainX, trainY)


# prepare the k-fold cross-validation configuration
n_folds = 10
kfold = KFold(n_folds, True, 1)
X=trainX#np.arange(training3D_matrix,testing3D_matrix)
print(X)
y=trainY#np.arange(trainY,testY)
# cross validation estimation of performance
scores, members = list(), list()
for train_ix, test_ix in kfold.split(X):
        trainX, trainy = X[train_ix], y[train_ix]
        testX, testy = X[test_ix], y[test_ix]
        mode = train_model(trainX, trainy)
        test_acc = test_model(mode,testX, testy)
        print(test_acc)
        scores.append(test_acc)
        members.append(mode)
sco = []
for row in scores:
    sco.append(row)
print(mean(sco))
    