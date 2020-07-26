#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:07:57 2019

@author: mazzone
"""

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

np.random.seed(500)

class_weight = {0: 1.8,
                1: 3.2,
                2: 1.,
                3: 1.2}


le =  ce.OneHotEncoder(return_df=False, handle_unknown="ignore")
training3D_matrix = joblib.load('../dataPreProcessing/matrix3D_ANSWER_word2vecNOLEMMA')
trainingSet = pd.read_csv('trainingSet1165.csv', sep = ',')
cat=trainingSet['categoria']
data=np.array(trainingSet['domanda'])
data= trainingSet['domanda']
print(data)
#label_encoder = LabelEncoder()
#integer_encoded = label_encoder.fit_transform(cat)
# binary encode
#onehot_encoder = OneHotEncoder(sparse=False)
#integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
#onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


#test3D_matrix = we.wordEmbedding(frase)

gnb = GaussianNB()
print(cat.shape)
gnb.fit(training3D_matrix, cat)

#y_pred = gnb.predict (test3D_matrix)


y_TT = le.fit_transform(cat.tolist())

y_predTT = le.transform(y_pred.tolist())

roc = roc_auc_score ( y_TT , y_predTT )
roc_val = roc_auc_score ( y_TT , y_predTT )


print ( '\rroc-auc: %s - roc-auc_val: %s' % (str ( round ( roc , 4 ) ) , str ( round ( roc_val , 4 ) )) ,
                    end=100 * ' ' + '\n' )


print(classification_report(y_pred , '3' ))

acc = accuracy_score ( y_pred , '3' )
print ( " Acc:" , acc )

pre = precision_score ( y_pred , '3' , average='micro')
print ( " Pre:" , pre )

re = recall_score ( y_pred , '3' , average='micro')
print ( " Rec:" , re )

f1 = f1_score ( y_pred , '3' , average='micro')
print ( " F1:" , f1 )


#joblib.dump(m, "trained_model.dump")
