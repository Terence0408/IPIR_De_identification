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

import nltk
import re
import h5py # It needs at save keras model
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.engine.topology import Merge
from scipy import spatial


chars= [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
        'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

def what_d(dim=100, runtimes = 1, renew =True, maxlen=18):

    char_X_100 = 2196017               # words: 399488
    char_X_010 = maxlen               # max word length: 43
    char_X_001 = 37                   # chars: 37
    char_Y_10  = 2196017               # words: 399488
    char_Y_01  = dim                  # encode word length: 100

    # Read and arrange data set into x, y type.
    text_file = open(path+"glove.6B/glove.840B.300d.txt", 'r')
    glove = text_file.readlines()

    vocab = []
    X = np.zeros((char_X_100, char_X_010, char_X_001), dtype=np.bool)
    y = np.zeros((char_Y_10, char_Y_01 ), dtype=np.float64)

    ii = 0
    for i in range(0,2196017):
        lists = glove[i].split()
        lists[0] = re.sub("[^0-9a-zA-Z]", "", lists[0])
        if 0 < len(lists[0]) <= maxlen:
            print ii, i
            vocab.append(lists[0])
            text = lists[0].ljust(char_X_010)
            for j in range(0, char_X_010):
                X[ii, j, char_indices[text[j]]] = 1
            for k in range(1,dim+1):
                y[ii,k-1] = lists[k]
            ii = ii + 1
        #if i % 40000 == 0:
        #   print i

    # Find par.

    lens=[]
    for word in vocab:
        lens.append(len(word))
    print max(lens)
    print len(vocab) # 399488
    char_X_100 = len(vocab)
    char_Y_10  = len(vocab)
    X = X[0:len(vocab)]
    y = y[0:len(vocab)]

    # First time: build the model: a bidirectional LSTM
    if renew == True:
        print('Build model...')
        left = Sequential()
        left.add(SimpleRNN(dim, input_shape=(char_X_010, char_X_001), activation='tanh',
                      #inner_activation='sigmoid', dropout_W=0.5, dropout_U=0.5))
                      dropout_W=0.5, dropout_U=0.5))
        right = Sequential()
        right.add(SimpleRNN(dim, input_shape=(char_X_010, char_X_001), activation='tanh',
                       #inner_activation='sigmoid', dropout_W=0.5, dropout_U=0.5, go_backwards=True))
                       dropout_W=0.5, dropout_U=0.5, go_backwards=True))
        model = Sequential()
        model.add(Merge([left, right], mode='sum'))
        model.compile('Adadelta', 'MSE', metrics=['accuracy'])
        model.fit([X,X], y, batch_size=512, nb_epoch=1)
        model.save(path+"model/biRNN_char_pretrain_twitter_"+str(dim)+"d.pk")


    # Not first time: build the model: a bidirectional LSTM

    print('Load model...')
    model = load_model(path+"model/biRNN_char_pretrain_twitter_"+str(dim)+"d.pk")
    for j in range(0,runtimes-1):
        print('Build model...')
        model.fit([X,X], y,
                  batch_size=512,
                  nb_epoch=1)
        model.save(path+"model/biRNN_char_pretrain_twitter_"+str(dim)+"d.pk")


    # Test cosine similarity, train set

    cos = []
    for i in range(0, len(vocab)):
        text = vocab[i].ljust(char_X_010)
        x = np.zeros((1, char_X_010, char_X_001), dtype=np.bool)
        for j in range(0, len(text)):
            x[0, j, char_indices[text[j]]] = 1
        map_LSTM = model.predict([x, x], verbose=0)

        map_GloVe = y[i]

        cos.append(1 - spatial.distance.cosine(map_LSTM, map_GloVe))
    f = open(path+"model/cosine.txt", 'a')
    f.write(str(dim)+"d. 22 times biRNN twitter cosine similarity: "+str(sum(cos)/len(cos))+"\n")
    f.close()


    # Test cosine similarity, misspelling

    cos = []
    change_engs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                   'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    for i in range(0, len(vocab)):
        misspelling = vocab[i]
        if len(misspelling)>4:
            loc = int(np.random.uniform(0,1,1)*len(misspelling))
            cha = int(np.random.uniform(0,1,1)*26)

            tem = list(misspelling)
            tem[loc] = change_engs[cha]

            misspelling = "".join(tem)
            text = misspelling.ljust(char_X_010)
            x = np.zeros((1, char_X_010, char_X_001), dtype=np.bool)
            for j in range(0, len(text)):
                x[0, j, char_indices[text[j]]] = 1
            map_LSTM = model.predict([x, x], verbose=0)

            map_GloVe = y[i]

            cos.append(1 - spatial.distance.cosine(map_LSTM, map_GloVe))
    f = open(path+"model/cosine.txt", 'a')
    f.write(str(dim)+"d. 22 times biRNN twitter misspelling cosine similarity : "+str(sum(cos)/len(cos))+", len: "+str(len(cos))+"\n")
    f.close()

what_d(dim=300, runtimes =  22, renew =True,  maxlen = 18)

print "end"


