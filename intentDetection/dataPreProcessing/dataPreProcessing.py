#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 19:27:29 2019

@author: mazzone
"""
import numpy as np
import pandas as pd
import joblib
#import word2vecEmbedding as we
import fastTextEmbedding as we


trainingSet = pd.read_csv('trainingSet1165.csv', sep = ',')
testingSet = pd.read_csv('testingSet.csv', sep = ';')

training=np.array(trainingSet['domanda'])
testing=np.array(testingSet['domanda'])
trainingmatrix3D = np.zeros((len(training),50,300))
testingmatrix3D = np.zeros((len(testing),50,300))

trainingmatrix3D = we.wordEmbedding(training)
testingmatrix3D = we.wordEmbedding(testing)



#joblib.dump(trainingmatrix3D, 'matrix3D_TRAININGANSWER_fastTextNOLEMMA')
#joblib.dump(testingmatrix3D, 'matrix3D_TESTINGANSWER_fastTextNOLEMMA')
