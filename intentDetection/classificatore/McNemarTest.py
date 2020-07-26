#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:10:37 2019

@author: mazzone
"""
import joblib


import numpy as np
import pandas as pd
from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar
import statsmodels.stats.contingency_tables

def mcNemar(target,model1,model2):
    y_target = np.array(target)
    # Class labels predicted by model 1
    y_model1 = np.array(model1)
    # Class labels predicted by model 2
    y_model2 = np.array(model2)
    tb = mcnemar_table(y_target=y_target, 
                       y_model1=y_model1, 
                       y_model2=y_model2)
    #print (tb)
    return tb

result=[]
result.append(joblib.load('resultW2VNLBiLSTM'))
result.append(joblib.load('resultW2VNLCLSTM'))
result.append(joblib.load('resultW2VNLCBiLSTM'))
result.append(joblib.load('resultFTNLBiLSTM'))
result.append(joblib.load('resultFTNLCLSTM'))
result.append(joblib.load('resultFTNLCBiLSTM'))
result.append(joblib.load('resultSVM'))

# The correct target (class) labels
testSet = pd.read_csv('testingSet.csv', sep = ';')
testSetY= np.array(testSet['categoria'])
y_target = []
for n in testSetY:
    y_target.append(int(n) - 1)

matrix2D = np.zeros((7,7))
for index in range (0,7):
    for indey in range (0,7):
        if not index==indey:
            table = mcNemar(y_target,result[index],result[indey])
            print(index,indey,table)
            p = statsmodels.stats.contingency_tables.mcnemar(table, exact=True, correction=True)
            print(p.pvalue)
            matrix2D[index][indey] = round(p.pvalue, 3)
        else: 
            matrix2D[index][indey] = 0
print(matrix2D)


# Creates pandas DataFrame. 
#d = {'one' : pd.Series([10, 20, 30, 40], index =['a', 'b', 'c', 'd']), 
#      'two' : pd.Series([10, 20, 30, 40], index =['a', 'b', 'c', 'd'])} 
  
# creates Dataframe. 
#df = pd.DataFrame() 

df = pd.DataFrame(matrix2D, index =['W2VNLBiLSTM', 'W2VNLCLSTM', 'W2VNLCBiLSTM', 'FTNLBiLSTM', 'FTNLCLSTM', 'FTNLCBiLSTM', 'SVM'], columns =['W2VNLBiLSTM', 'W2VNLCLSTM', 'W2VNLCBiLSTM', 'FTNLBiLSTM', 'FTNLCLSTM', 'FTNLCBiLSTM', 'SVM']) 

print(df)

joblib.dump(df.to_csv(),"mcnemar.csv")
