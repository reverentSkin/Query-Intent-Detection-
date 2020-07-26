#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 19:27:29 2019

@author: mazzone
"""

import pandas as pd
import joblib

#import fastTextEmbedding as we
import tfidfVectorization as tfidf
from sklearn import svm
from sklearn.metrics import accuracy_score

trainingSet = pd.read_csv('../dataPreProcessing/trainingSet1165.csv', sep = ',')
testingSet = pd.read_csv('../dataPreProcessing/testingSet.csv', sep = ';')

#model = tfidf.tfidf(trainingSet['domanda'])
#model = joblib.load('tfidf')

#trainX= tfidf.dataEmbedding(model,trainingSet)
#testX= tfidf.dataEmbedding(model,testingSet)
trainX = joblib.load('../dataPreProcessing/trainingTFIDF')
testX = joblib.load('../dataPreProcessing/testingTFIDF')

trainY=tfidf.encodeCategory(trainingSet)
testY=tfidf.encodeCategory(testingSet)

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(trainX,trainY)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(testX)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, testY)*100)



#joblib.dump(SVM, 'SVMmodelv6')



#print(model.similarity("sporco","sporco"))
# print(index, tokIndex)




