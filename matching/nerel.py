#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:51:39 2019

@author: mazzone
"""
import spacy
import json
import it_core_news_sm
import numpy as np
from spacy.pipeline import EntityRecognizer
from PIL import Image
import requests
from io import BytesIO
from dandelion import DataTXT
from googletrans import Translator

def nerelEn(text):
    #text="voglio andare in bici. Che percorso mi consigliate?"
    translator = Translator()
    tr=translator.translate(text)
    text=tr.text
    datatxt = DataTXT(app_id='5cb879ebda544e2e95ce5cefd4963aca', app_key='5cb879ebda544e2e95ce5cefd4963aca')
    response = datatxt.nex(text, min_confidence=0.20, include_types=True, include_abstract=True, include_lod=True, include_categories=True)
    time = response['annotations']
    #print(time)
    #entity = []
    #print(time)
    index=0
    categories=[]
    entity = [] 
    types=[]
    lods=[]
    for index, row in enumerate(time):

        ca=[]
        ty=[]
        lo=[]
        name = time[index]['spot']
        entity.append(name)
        try:
            categoria = time[index]['categories']
            ca.append(categoria)
            for r in ca:
                for o in r:
                    categories.append(o)
        except:
            print('categories not present')
            #categories.append("")

        try:
            typ = time[index]['types']
            ty.append(typ)
            for r in ty:
                for o in r:
                    types.append(o)
        except:
            print('types not present')
            #types.append("")
        try:
            lod = time[index]['lod']['dbpedia']
            lo.append(lod)
            for r in lo:
                    lods.append(r)
        except:
            print('lod not present')

        #print(lo)
    return (text,entity,categories,types,lods)

#text="voglio andare in bici. Che percorso mi consigliate?"
def nerel(text):
    datatxt = DataTXT(app_id='5cb879ebda544e2e95ce5cefd4963aca', app_key='5cb879ebda544e2e95ce5cefd4963aca')
    response = datatxt.nex(text, min_confidence=0.20, include_abstract=True, include_confidence=True, include_categories=True, include_image=True)
    time = response['annotations']

    mostConfidence=0
    #print(response)
    index=0
    entity=[]
    abstracts=[]
    confidences=[]
    mostConf=0
    mostimage=""
    categories=[]
    for index, row in enumerate(time):

        ca=[]
        name = time[index]['spot']
        entity.append(name)
        try:
            abstract = time[index]['abstract']
            abstracts.append(abstract)
            #print(abstract)
        except:
            print('abstract not present')
            abstracts.append("abstact not present")
        try:
            confidence = time[index]['confidence']
            if confidence > mostConfidence:
                #print('ok')
                mostConfidence=confidence
                mostConf=name
                mostimage=time[index]['image']['thumbnail']
            #print(confidence)
            confidences.append(confidence)
        except:
            print('confidence not present')
            confidences.append("")
        try:
            categoria = time[index]['categories']
            ca.append(categoria)
            for r in ca:
                for o in r:
                    #print(o)
                    categories.append(o)
        except:
            print('categories not present')
            #categories.append("")


    return (entity, abstracts, confidences, categories, mostConf, mostimage)
