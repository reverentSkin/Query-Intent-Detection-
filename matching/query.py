#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:18:37 2019

@author: mazzone
"""
import pandas as pd
import nerel as nl
import numpy as np
import joblib
from numpy import argmax
import word2vecEmbedding as we
import psycopg2



def Invert(x):
    inverted = argmax(x)
    return(inverted)

def intentDetection(text):
    model = joblib.load('word2vec_NOlemC-BiLSTMv6') 
    w2v = we.wordEmbedding(text)
    result = model.predict(w2v)
    result = Invert(result)
    return result+1

def topicRecognation(frase):
    return 0


def queryRappresentation(frase):
    query = []
    abstr="No abstract."
    #frase="devo aggiustare i miei vestiti... c'è una sarta che può aiutarmi??"
    query.append(frase)
    
    intent = intentDetection(query)
    query.append(str(intent))
    topic = topicRecognation(frase)
    query.append(str(topic))
    entita, abstracts, confidences, categorie, mostConfidence, mostimage= nl.nerel(frase)
    query.append(entita)
    query.append(categorie)
    query.append(str(confidences))
    query.append(mostConfidence)
    query.append(mostimage)
    query.append(abstracts)
    text, entity,categories,types,lods = nl.nerelEn(frase)
    query.append(categories)
    query.append(types)
    query.append(lods)
    if ' ' in entity:
        entity = entity.split(' ')
        for r in entity:
            query.append(entity)
    else:
        query.append(entity)
    
    print(query)
    queryDB(query)
    #print(query[9])
    for index, i in enumerate(entity):
        if i == mostConfidence:
            abstr=abstracts[index]
    return query, abstr
    print(query)
    
def queryDB(q):
    try:  #connessione al database.
       connection = psycopg2.connect(user = "postgres",
                                     password = "Giuppy97",
                                     host = "localhost",
                                     port = "5432",
                                     database = "users")
       
       cursor = connection.cursor()
       
       stri = 'INSERT INTO public.query(testo, intent, topic, entita, categorie, confidences, "mostConfidence", "mostImage", abstract, categories, types, lod,entity) '+"VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)" 

       cursor.execute(stri, (q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12]))
       connection.commit()
       
           
    except (Exception, psycopg2.Error) as error :
        print ("Error while connecting to PostgreSQL", error)
        return 0
           
    finally:
    #Chiusura della connessione.
        if(connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
            return 1
       

       