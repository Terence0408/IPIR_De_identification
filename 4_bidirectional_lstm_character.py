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


from keras.models import Sequential
from keras.layers import LSTM
from keras.engine.topology import Merge



vocab = pickle.load(open(path+"model/GloVe_vocab.pk", "rb" ))
W = pickle.load(open(path+"model/GloVe_W.pk", "rb" ))

#if "having" in vocab.keys():
#    print W[vocab['having'][0]]




chars = sorted(list(set(" ".join(vocab.keys()))))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


len_X_100 = len(vocab.keys())                # words: 30642
len_X_010 = len(max(vocab.keys(), key=len))  # max word length: 32
len_X_001 = len(chars)                       # chars: 37
len_Y_10  = W.shape[0]                       # words: 30642
len_Y_01  = W.shape[1]                       # encode word length: 100


X = np.zeros((len_X_100, len_X_010, len_X_001), dtype=np.bool)
y = np.zeros((len_Y_10, len_Y_01 ), dtype=np.float64)

for i in range(0, len_X_100):
    text = vocab.keys()[i].ljust(len_X_010)
    for j in range(0, len_X_010):
        X[i, j, char_indices[text[j]]] = 1
    y[i] = W[vocab[vocab.keys()[i]][0]]

'''
# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(100, input_shape=(len_X_010, len_X_001) ,activation='tanh', inner_activation='sigmoid', dropout_W=0.5, dropout_U=0.5))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

model.fit(X, y,
          batch_size=128,
          nb_epoch=1)
'''

# build the model: a single LSTM
print('Build model...')
left = Sequential()
left.add(LSTM(100, input_shape=(len_X_010, len_X_001), activation='tanh',
              inner_activation='sigmoid', dropout_W=0.5, dropout_U=0.5))

right = Sequential()
right.add(LSTM(100, input_shape=(len_X_010, len_X_001), activation='tanh',
               inner_activation='sigmoid', dropout_W=0.5, dropout_U=0.5, go_backwards=True))


model = Sequential()
model.add(Merge([left, right], mode='sum'))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

model.fit([X,X], y,
          batch_size=128,
          nb_epoch=1)


preds = model.predict([X[0:1],X[0:1]], verbose=0)[0]


print "end"