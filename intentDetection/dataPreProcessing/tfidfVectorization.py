#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:41:29 2019

@author: mazzone
"""
import random as rn
import numpy as np


import joblib
from spacy.symbols import NUM, AUX, VERB
import it_core_news_sm
import pandas as pd
import numpy as np


from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

def tokenization(frase):
    nlp=it_core_news_sm.load()
    tok = list()
    row = list()
    stop_word = ['e','o','','il','lo','la','un','uno','una','mia','mio','tuo','con','su','per','tra','fra','please','aiuto','urgente','help',',','.','..','...','....',':',';','!','(',')','-',' ','/','"']
    sentence = np.zeros(50)
    frase= frase.lower()
    print("domanda = ", frase)
    sentence = nlp(frase)

    vec =[]

    for tok in sentence:
        word = tok.text.lower()
        flag = True
        #delete stopWords
        if word in stop_word:
            flag = False
        #delete numbers
        elif tok.pos == NUM:
            flag = False

        elif tok.pos == VERB or tok.pos == AUX:
            word=tok.lemma_

        if flag==True:
            vec.append(tok.text)
    return vec


def dataEmbedding(model, data):
    for indexs, row in enumerate(data['domanda']):
        g_vec = tokenization(row)
        print(indexs)
        data.loc[indexs,'domanda'] = str(g_vec)

    X_Tfidf = model.transform(data['domanda'])
    return X_Tfidf


def tfidf(data):
    Tfidf_vect = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
    Tfidf_vect.fit(data)
    print(Tfidf_vect.vocabulary_)
    return Tfidf_vect

def encodeCategory(data):
    Y = np.array(data['categoria'])
    Encoder = LabelEncoder()
    Y = Encoder.fit_transform(Y)
    return Y
