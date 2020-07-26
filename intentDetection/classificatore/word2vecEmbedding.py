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
from gensim.models import Word2Vec

#get_file(
      #fname="skipgram_wiki_window10_size300_neg-samples10.tar.gz", 
      #origin="http://hlt.isti.cnr.it/wordembeddings/skipgram_wiki_window10_size300_neg-samples10.tar.gz", 
      #extract=True)


def wordEmbedding(data):
    
    
    model = Word2Vec.load('word2vec/common/word2vec/models/wiki_iter=5_algorithm=skipgram_window=10_size=300_neg-samples=10.m') # word2vec model
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
        
        frase= row.lower()
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
                if model.wv.__contains__(word):
                    g_vec = model.wv.__getitem__(word)
                    goodToken += 1
                    #print(g_vec[:300])
                 
                #set a random word embedding fon unknown words
                else:
                    badToken += 1
                    g_vec = model.wv.__getitem__(rn.choice(model.wv.index2entity))
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
    print('Il vocabolario è: ', len(model.wv.vocab))
    print('le domande trovate sono:', nAns)
    
    print(matrix3D.shape)
    
    return matrix3D