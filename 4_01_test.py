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

from scipy import spatial


vocab = pickle.load(open(path+"model/GloVe_vocab.pk", "rb" ))
W = pickle.load(open(path+"model/GloVe_W.pk", "rb" ))



chars= [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
        'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

char_X_100 = len(vocab.keys())                # words: 33299
char_X_010 = len(max(vocab.keys(), key=len))  # max word length: 54
char_X_001 = len(chars)                       # chars: 37
char_Y_10  = W.shape[0]                       # words: 33299
char_Y_01  = W.shape[1]                       # encode word length: 100


cos = []
model = load_model(path+"model/biLSTM_char.pk")
for word in vocab.keys():
    text = word.ljust(char_X_010)
    x = np.zeros((1, char_X_010, char_X_001), dtype=np.bool)

    for j in range(0, len(text)):
        x[0, j, char_indices[text[j]]] = 1
    map_LSTM = model.predict([x, x], verbose=0)

    map_GloVe = W[vocab[word][0]]

    cos.append(1 - spatial.distance.cosine(map_LSTM, map_GloVe))

print sum(cos)/len(cos)
