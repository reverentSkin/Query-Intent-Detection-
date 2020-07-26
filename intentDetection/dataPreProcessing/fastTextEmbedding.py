#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:41:29 2019

@author: mazzone
"""
import random as rn
#import connectionDB as eA
import numpy as np


import joblib
from spacy.symbols import NUM
import it_core_news_sm
import gensim as gs
from gensim.models import FastText
from gensim.test.utils import datapath


fastTextPreTrained = get_file(fname="cc.it.300.bin", origin="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.it.300.bin.gz", extract=True)





def wordEmbedding(data):
    cap_path = datapath('cc.it.300.bin')
    modelfast = gs.models.fasttext.load_facebook_vectors(cap_path)
    nlp=it_core_news_sm.load()
    stop_word = ['e','o','','il','lo','la','un','uno','una','mia','mio','tuo','con','su','per','tra','fra','please','aiuto','urgente','help',',','.','..','...','....',':',';','!','(',')','-',' ','/','"']


    tok = list()
    row = list()
    sentence = np.zeros(50)
    matrix3D = np.zeros((len(data),50,300))
    indexs=0
    badToken=0
    goodToken=0
    nAns=0
    for row in data:

        frase = row.lower()
        print("domanda = ", frase)
        sentence = nlp(frase)
        tokIndex=0
        nAns += 1

        for tok in sentence:
            print([(tok.text, tok.pos_) ])
            #word = tok.lemma_
            word = tok.text.lower()
            g_vec =[]
            flag = True

            #delete stopWords
            if word in stop_word:
                flag = False

            #delete numbers
            elif tok.pos == NUM:
                flag = False

            elif word == 'è':
                word=tok.lemma_

            if flag==True:
                if modelfast.wv.__contains__(word):
                    g_vec = modelfast.wv.__getitem__(word)
                    goodToken += 1
                    #print(g_vec[:300])

                #set a random word embedding fon unknown words
                else:
                    badToken += 1
                    g_vec = modelfast.wv.__getitem__(rn.choice(modelfast.wv.index2entity))
                    #print(g_vec [:300])


            if flag==True:
            # VECTOR FOR EACH TOKEN
                tok2D = []
                tok2D = g_vec
                #print(tok2D[:300])
                if tokIndex < 50:
                    matrix3D[indexs][tokIndex] = tok2D
                tokIndex += 1
        indexs += 1

    print ('I token trovati sono:' ,goodToken)
    print ('I token non trovati sono:' , badToken)
    print('Il vocabolario è: ', len(modelfast.wv.vocab))
    print('le domande trovate sono:', nAns)

    print(matrix3D.shape)

    return matrix3D
