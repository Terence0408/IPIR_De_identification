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
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file


vocab = pickle.load(open(path+"model/GloVe_vocab.pk", "rb" ))
W = pickle.load(open(path+"model/GloVe_W.pk", "rb" ))

if "having" in vocab.keys():
    print W[vocab['having'][0]]

maxlen = len(max(vocab.keys(), key=len))
# maxlen = 32

chars = sorted(list(set(" ".join(vocab.keys()))))


len("123".ljust(maxlen))



X = np.zeros((len(vocab.keys()), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((W.shape[0], W.shape[1]), dtype=np.float64)

for i in vocab.keys():
    texts.append(i.ljust(maxlen))









print "end"