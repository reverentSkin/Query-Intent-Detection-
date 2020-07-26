o#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 22:00:08 2019

@author: mazzone
"""

import joblib
import pandas as pd
from numpy import argmax
import numpy as np
#from numpy import argmax
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def invert(x):
    result = []
    for row in x:
        inverted = argmax(row)
        result.append(inverted)
    return(result)  
    



w2vLSTM = joblib.load('word2vec_LSTM') 
#w2vBiLSTM = joblib.load('word2vec_BiLSTM') 
w2vCLSTM = joblib.load('word2vec_C-LSTMv4') 
#w2vCBiLSTM = joblib.load('word2vec_C-BiLSTM') 
w2vNLLSTM = joblib.load('word2vec_NOlemLSTMv6') 
w2vNLBiLSTM = joblib.load('word2vec_NOlemBiLSTMv7') 
w2vNLCLSTM = joblib.load('word2vec_NOlemC-LSTMv6') 
w2vNLCBiLSTM = joblib.load('word2vec_NOlemC-BILSTMv6') 

#fTLSTM = joblib.load('fastText_LSTM')
#fTBiLSTM = joblib.load('fastText_BiLSTM')
#fTCLSTM = joblib.load('fastText_C-LSTM')
#fTCBiLSTM = joblib.load('fastText_C-LSTM')
fTNLLSTM = joblib.load('fastText_NOlemLSTMv7')
fTNLBiLSTM = joblib.load('fastText_NOlemBiLSTMv7')
fTNLCLSTM = joblib.load('fastText_NOlemC-LSTMv7')
fTNLCBiLSTM = joblib.load('fastText_NOlemC-BiLSTMv6')

#fidf_vect = joblib.load('tf-idf')

Naive = joblib.load('NaiveModelv6')

SVM = joblib.load('SVMmodelv6')

testSet = pd.read_csv('../dataPreProcessing/testingSet.csv', sep = ';')
testSetY= np.array(testSet['categoria'])

#testSetX = ["Ho bisogno di un consiglio da parte di una persona del posto", 
 #        "Aiuto ho bisogno di un professore per imparare a suonare il flauto",
  #       "Cosa visitare a Bari nel pomeriggio?",
  #       "Quali chiese visitare vicino a me?",
  #       "c'è un negozio di articoli da giardino?",
  #       "Dove posso comprare un profumo Armani?",
  #       "Sto cercando la villa comunale... è presente una villa in cui sdraiarsi?",
  #       "non capisco il dialetto... qualcuno può aiutarmi?",
  #       "cercasi informazioni su chiese?",
  #       "c'è un agricoltore disponibile?",
  #       "sono appena arrivato a Bari... che cosa posso fare?",
  #       "eventi culturali nella zona?",
  #       "dove posso trovare un buon veterinario per il mio cane??",
  #       "c'è un agricoltore che vende frutta senza conservanti?",
  #
  #      "Dove posso comprare un computer Acer a Martina?",
  #       "dove posso trovare un negozio che vende le magliette da donna?"]
#testSetY= [2, 1, 3, 3, 0, 0, 2, 2, 2, 1, 3, 2, 1, 1, 0, 0]

data=joblib.load('../dataPreProcessing/matrix3D_TESTINGANSWER_word2vecNOLEMMA')

res= w2vLSTM.predict(data)
result1= invert(res)

res= w2vCLSTM.predict(data)
result2= invert(res)

res= w2vNLLSTM.predict(data)
result3= invert(res)

res= w2vNLBiLSTM.predict(data)
result4= invert(res)
joblib.dump(result4, 'resultW2VNLBiLSTM')

res= w2vNLCLSTM.predict(data)
result5= invert(res)
joblib.dump(result5, 'resultW2VNLCLSTM')

res= w2vNLCBiLSTM.predict(data)
result6= invert(res)
joblib.dump(result6, 'resultW2VNLCBiLSTM')

print(result3)
data=joblib.load('../dataPreProcessing/matrix3D_TESTINGANSWER_fastTextNOLEMMA')

res= fTNLLSTM.predict(data)
result7= invert(res)

res= fTNLBiLSTM.predict(data)
result8= invert(res)
joblib.dump(result8, 'resultFTNLBiLSTM')

res= fTNLCLSTM.predict(data)
result9= invert(res)
joblib.dump(result9, 'resultFTNLCLSTM')

res= fTNLCBiLSTM.predict(data)
result10= invert(res)
joblib.dump(result10, 'resultFTNLCBiLSTM')

data = joblib.load('../dataPreProcessing/testingTFIDF')

result11 = Naive.predict(data)


result12 = SVM.predict(data)
joblib.dump(result12, 'resultSVM')
index=0
for n in testSetY:
    testSetY[index] = int(n) - 1
    index += 1
print (testSetY)
# evaluate the model
accuracy1 = accuracy_score(testSetY, result1)
accuracy2 = accuracy_score(testSetY, result2)
accuracy3 = accuracy_score(testSetY, result3)
accuracy4 = accuracy_score(testSetY, result4)
accuracy5 = accuracy_score(testSetY, result5)
accuracy6 = accuracy_score(testSetY, result6)
accuracy7 = accuracy_score(testSetY, result7)
accuracy8 = accuracy_score(testSetY, result8)
accuracy9 = accuracy_score(testSetY, result9)
accuracy10 = accuracy_score(testSetY, result10)
accuracy11 = accuracy_score(testSetY, result11)
accuracy12 = accuracy_score(testSetY, result12)

precision1, recall1, f11, z = precision_recall_fscore_support(testSetY, result1, average='macro')
precision2, recall2, f12, z = precision_recall_fscore_support(testSetY, result2, average='macro')
precision3, recall3, f13, z = precision_recall_fscore_support(testSetY, result3, average='macro')
precision4, recall4, f14, z = precision_recall_fscore_support(testSetY, result4, average='macro')
precision5, recall5, f15, z = precision_recall_fscore_support(testSetY, result5, average='macro')
precision6, recall6, f16, z = precision_recall_fscore_support(testSetY, result6, average='macro')
precision7, recall7, f17, z = precision_recall_fscore_support(testSetY, result7, average='macro')
precision8, recall8, f18, z = precision_recall_fscore_support(testSetY, result8, average='macro')
precision9, recall9, f19, z = precision_recall_fscore_support(testSetY, result9, average='macro')
precision10, recall10, f110, z = precision_recall_fscore_support(testSetY, result10, average='macro')
precision11, recall11, f111, z = precision_recall_fscore_support(testSetY, result11, average='macro')
precision12, recall12, f112, z = precision_recall_fscore_support(testSetY, result12, average='macro')


tabel= pd.DataFrame({"model":['w2v LSTM','w2v C-LSTM','w2v  nolem LSTM','w2v nolem BiLSTM','w2v nolem C-LSTM','w2v nolem C-BiLSTM','FT nolem LSTM','FT nolem BiLSTM','FT nolem C-LSTM','FT nolem C-BiLSTM','NAIVE','SVM'],
                     "accuracy":[accuracy1,accuracy2,accuracy3,accuracy4,accuracy5,accuracy6,accuracy7,accuracy8,accuracy9,accuracy10,accuracy11,accuracy12],
                     "precision":[precision1,precision2,precision3,precision4,precision5,precision6,precision7,precision8,precision9,precision10,precision11,precision12],
                     "recall":[recall1,recall2,recall3,recall4,recall5,recall6,recall7,recall8,recall9,recall10,recall11,recall12],
                     "f1":[f11,f12,f13,f14,f15,f16,f17,f18,f19,f110,f111,f112]})
print (tabel)

    
