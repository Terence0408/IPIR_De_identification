#!/usr/bin/python
# -*- coding: utf-8 -*-

# Goal: Build Character-enhanced token embedding layer to encode token.
# Step: 1. Transfer character into 25 binary vector. If all vectors are 0, this character present 'a'.
#       2. Build bidirectional lstm layer.
#             Input: character(25 binary vector) sequence of a token.
#             Output: 100 dimension.
#          Train step: x: character(25 binary vector) sequence of a token.
#                      y: token correspond to GloVe result.
#                      Dropout: prob. = 0.5
#       3. Concentrate GloVe result and lstm result. If no GloVe result, use lstm result instant of.

import logging
import cPickle as pickle
import psycopg2
import numpy as np
np.random.seed(19870712)  # for reproducibility
path = "/home/terence/pycharm_use/IPIR_De_identification/1_data/"
get_conn = psycopg2.connect(dbname='IPIR_De_identification',user='postgres', host='localhost', password='postgres')

import h5py # It needs at save keras model
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.engine.topology import Merge



vocab = pickle.load(open(path+"model/GloVe_vocab.pk", "rb" ))
W = pickle.load(open(path+"model/GloVe_W.pk", "rb" ))

#load model and test.
'''
if "having" in vocab.keys():
   print W[vocab['having'][0]]
print 'Load no error.'
'''

chars= [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
        'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


char_X_100 = len(vocab.keys())                # words: 30642
char_X_010 = len(max(vocab.keys(), key=len))  # max word length: 32
char_X_001 = len(chars)                       # chars: 37
char_Y_10  = W.shape[0]                       # words: 30642
char_Y_01  = W.shape[1]                       # encode word length: 100


X = np.zeros((char_X_100, char_X_010, char_X_001), dtype=np.bool)
y = np.zeros((char_Y_10, char_Y_01 ), dtype=np.float64)

for i in range(0, char_X_100):
    text = vocab.keys()[i].ljust(char_X_010)
    for j in range(0, char_X_010):
        X[i, j, char_indices[text[j]]] = 1
    y[i] = W[vocab[vocab.keys()[i]][0]]



# build the model: a bidirectional LSTM
print('Build model...')
left = Sequential()
left.add(LSTM(100, input_shape=(char_X_010, char_X_001), activation='tanh',
              inner_activation='sigmoid', dropout_W=0.5, dropout_U=0.5))

right = Sequential()
right.add(LSTM(100, input_shape=(char_X_010, char_X_001), activation='tanh',
               inner_activation='sigmoid', dropout_W=0.5, dropout_U=0.5, go_backwards=True))


model = Sequential()
model.add(Merge([left, right], mode='sum'))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

model.fit([X,X], y,
          batch_size=128,
          nb_epoch=1)

model.save(path+"model/biLSTM_char.pk")
print "end"

# load model and test.

'''
model = load_model(path+"model/biLSTM_char.pk")
chars= [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
        'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

if "having" in vocab.keys():
    map_GloVe = W[vocab['having'][0]]

text = 'having'.ljust(len_X_010)
x = np.zeros((1, len_X_010, len_X_001), dtype=np.bool)

for j in range(0, len(text)):
    x[0, j, char_indices[text[j]]] = 1
map_LSTM = model.predict([x,x], verbose=0)
'''



