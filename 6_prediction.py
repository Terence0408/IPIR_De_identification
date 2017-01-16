#!/usr/bin/python
# -*- coding: utf-8 -*-


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
from keras.layers import LSTM, TimeDistributed, Dense, MaxoutDense
from keras.engine.topology import Merge

get_conn.autocommit = True
get_cur  = get_conn.cursor()


chars = [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
        'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

label = ["NONE", "BIOID", "DOCTOR", "ORGANIZATION", "PATIENT", "USERNAME",
         "FAX", "HEALTHPLAN", "HOSPITAL", "STREET", "AGE", "EMAIL",
         "PHONE", "DEVICE", "CITY", "DATE", "IDNUM", "URL",
         "COUNTRY", "LOCATION-OTHER", "STATE", "PROFESSION", "MEDICALRECORD", "ZIP"]
labe_indices = dict((c, i) for i, c in enumerate(label))
indices_labe = dict((i, c) for i, c in enumerate(label))

chara_x_100 = 1                                # words: 1
chara_x_010 = 54 # max word length: 54
chara_x_001 = len(chars)                       # chars: 37
chara_y_10  = 1                                # words: 1
chara_y_01  = 100                      # encode word length: 100

token_x_100 = 33700                 # sentences: 33700
token_x_010 = 30                               # sentences length: 30
token_x_001 = 100                    # encode word length: 100
token_y_100 = 33700                 # sentences: 33700
token_y_010 = 30                               # sentences length: 30
token_y_001 = 24                               # PHI level: 23+1


token_x = pickle.load(open(path+"model/token_x.pk", "rb" ))
token_y = pickle.load(open(path+"model/token_y.pk", "rb" ))
label_model = load_model(path+"model/biLSTM_label.pk")

pred = label_model.predict([token_x[0:1],token_x[0:1]], verbose=0)
pred_labels=[]
for j in range(0,token_y_010):
    pred_labels.append(indices_labe[np.argmax(pred[0][j])])

print "end"
