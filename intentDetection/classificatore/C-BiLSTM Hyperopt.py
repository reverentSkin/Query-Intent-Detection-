#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 09:34:57 2019

@author: mazzone
"""

# Keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Bidirectional, GlobalMaxPool1D
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import joblib

from sklearn.manifold import TSNE
from hyperopt import fmin, hp, tpe
from hyperopt import Trials, STATUS_OK
from keras.optimizers import Adam, SGD
from keras import backend as K

opt_search_space = hp.choice('name',
        [
        {'name': 'adam', 
            'learning_rate': hp.loguniform('learning_rate_adam', -10, 0), # Note the name of the label to avoid duplicates
        },
        {'name': 'sgd',
            'learning_rate': hp.loguniform('learning_rate_sgd', -15, 1), # Note the name of the label to avoid duplicates
            'momentum': hp.uniform('momentum', 0, 1.0),
        }
        ])

search_space = {
    'optimizer': opt_search_space,
    'layer_conv_size' : hp.quniform('layer_conv_size', low=100, high=513, q=50),#hp.choice('layer_conv_size', 2 ** np.arange(10)),
    
    'layer_LSTM_size': hp.choice('layer_LSTM_size', np.arange(100, 401, 50)),
    'batch_size':  hp.choice('batch_size', np.array(2 ** np.arange(7))),
    'dropout' : hp.choice('dropout', np.arange(0.3, 1, 0.1))
    
}

def create_model(params):
    mod = Sequential()
    mod.add(Dropout(int(params['dropout'])))
    print('this',int(params['dropout']))
    #print(search_space['layer_conv_size'])
    print(int(params['layer_conv_size']))
    mod.add(Conv1D(int(params['layer_conv_size']), 4, activation='relu', input_shape=(50,300)))
    mod.add(MaxPooling1D(pool_size=4))
    mod.add(Bidirectional(LSTM(units=int(params['layer_LSTM_size']), activation ='relu', return_sequences = True)))
    mod.add(GlobalMaxPool1D())
    mod.add(Dense(params['layer_LSTM_size'], activation='relu'))
    mod.add(Dropout(search_space['dropout']))
    mod.add(Dense(4, activation='softmax'))
    if params['optimizer']['name'] == 'adam':
        opt = Adam(lr=params['optimizer']['learning_rate'])
    else:
        opt = SGD(lr=params['optimizer']['learning_rate'], momentum=params['optimizer']['momentum'])
    mod.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return mod

def train_model(Xtrain, Ytrain, params):
    model = create_model(params)
    model.fit(training3D_matrix, trainY, validation_data=(testing3D_matrix, testY),batch_size=int(param['batch_size']), epochs=20, class_weight='auto')
    return model

def test_model(model, Xtest, YTest):
    
    test_accuracy = model.evaluate(Xtest, YTest)
    return test_accuracy


def hyperopt_fcn(params):
    model = train_model(training3D_matrix, trainY, params)
    test_acc = test_model(model, testing3D_matrix, testY)
    K.clear_session()
    return {'loss': -test_acc[1], 'status': STATUS_OK}
    
training3D_matrix = joblib.load('../dataPreProcessing/matrix3D_TRAININGANSWER_word2vecNOLEMMA')
testing3D_matrix = joblib.load('../dataPreProcessing/matrix3D_TESTINGANSWER_word2vecNOLEMMA')
#training3D_matrix = joblib.load('../dataPreProcessing/matrix3D_TRAININGANSWER_fastTextNOLEMMA')
#testing3D_matrix = joblib.load('../dataPreProcessing/matrix3D_TESTINGANSWER_fastTextNOLEMMA')
trainingSet = pd.read_csv('../dataPreProcessing/trainingSet1165.csv', sep = ',')
testingSet = pd.read_csv('../dataPreProcessing/testingSet.csv', sep = ';')
cat1=np.array(trainingSet['categoria'])
cat2=np.array(testingSet['categoria'])
label_encoder = LabelEncoder()
trainY = label_encoder.fit_transform(cat1)
testY= label_encoder.fit_transform(cat2)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
trainY = trainY.reshape(len(trainY), 1)
trainY = onehot_encoder.fit_transform(trainY)

testY = testY.reshape(len(testY), 1)
testY = onehot_encoder.fit_transform(testY)
    
best = fmin(hyperopt_fcn, search_space, algo=tpe.suggest, max_evals=200)

from hyperopt import space_eval
print(space_eval(search_space, best))
model = train_model(training3D_matrix, trainY, space_eval(search_space, best))
print('Accuracy is: ', test_model(model, testing3D_matrix, testY))

#model.fit(training3D_matrix, trainY, validation_data=(testing3D_matrix, testY), epochs=40, class_weight='auto')

#joblib.dump(model, 'word2vecNOlemC-BiLSTMv8')
#joblib.dump(model, 'fastTextNOlemC-BiLSTMv')