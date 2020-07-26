#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 19:27:29 2019

@author: mazzone
"""
import numpy as np
import pandas as pd
import joblib
import word2vecEmbedding as we
#import fastTextEmbedding as we
import tfidfVectorization as tfidf
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

trainingSet = pd.read_csv('../dataPreProcessing/trainingSet1165.csv', sep = ',')
testingSet = pd.read_csv('../dataPreProcessing/testingSet.csv', sep = ';')

#model = tfidf.tfidf(trainingSet['domanda'])
model = joblib.load('tfidf')

#trainX= tfidf.dataEmbedding(model,trainingSet)
#testX= tfidf.dataEmbedding(model,testingSet)
trainX = joblib.load('../dataPreProcessing/trainingTFIDF')
testX = joblib.load('../dataPreProcessing/testingTFIDF')

trainY=tfidf.encodeCategory(trainingSet)
testY=tfidf.encodeCategory(testingSet)


# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(trainX,trainY)
# predict the labels on validation dataset

predictions_NB = Naive.predict(testX)

# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, testY)*100)



joblib.dump(Naive, 'NaiveModelv6')



#print(model.similarity("sporco","sporco"))
# print(index, tokIndex)
