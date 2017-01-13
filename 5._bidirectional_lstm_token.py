#!/usr/bin/python
# -*- coding: utf-8 -*-

# Goal: Build label prediction layer and label sequence optimization layer to predict PHI type.
# Step: 1. Build bidirectional lstm layer.
#             Input: token(100 dimension) sequence of a sentence.
#             Output: 100 dimension.
#       2. Add dense layer
#             Input: 100 dimension(lstm layer result)
#             Output: 19 prob. dimension
#       3. Add label sequence optimization layer
#             Input: 19 prob. dimension(dense layer result)
#             Output: 19 PHI label(binary vector)
#       Train step: x: token(100 dimension) sequence of a sentence.
#                   y: 19 PHI label(18 PHI-type and 1 non PHI)


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
from keras.layers import LSTM, TimeDistributed, Dense
from keras.engine.topology import Merge

get_conn.autocommit = True
get_cur  = get_conn.cursor()

vocab = pickle.load(open(path+"model/GloVe_vocab.pk", "rb" ))
W = pickle.load(open(path+"model/GloVe_W.pk", "rb" ))
model = load_model(path+"model/biLSTM_char.pk")

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

# load model and test.
'''
if "having" in vocab.keys():
    map_GloVe = W[vocab['having'][0]]

text = 'having'.ljust(len_X_010)
x = np.zeros((1, len_X_010, len_X_001), dtype=np.bool)

for j in range(0, len(text)):
    x[0, j, char_indices[text[j]]] = 1
map_LSTM = model.predict([x,x], verbose=0)

print 'Load no error.'
'''










get_cur.execute("Select row_id, subject_id, order_id, content from record_text where train = 1;")# where subject_id= 253 and order_id =3;")
table = get_cur.fetchall()

# Split content into sentence.
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences=[]
for row in table:
    ter_query = "Select row_id, subject_id, order_id, text, id "
    ter_query += "from record_phi "
    ter_query += "where subject_id = " + str(row[1]) + " and order_id = " + str(row[2]) + " "
    ter_query += "order by id ;"
    get_cur.execute(ter_query)
    phi_table = get_cur.fetchall()

    content = row[3]
    for phi_row in phi_table:
        phi = re.sub(" ", "_", phi_row[3])
        phi_clean = re.sub("[^0-9a-zA-Z]", "", phi)
        content = re.sub(phi, phi_clean, content)
    row_sentences = tokenizer.tokenize(content)
    for sentence in row_sentences:
        sentences.append( " ".join(re.sub("[^0-9a-zA-Z]", " ", sentence).lower().split()))

print len(sentences) #33793


# Define sentence length 95% sentence less then 30 words.
'''
lens=[]
for i in sentences:
    lens.append(len(i.split()))
arlens=np.asarray(lens)
print np.percentile(arlens, 91.5) # 30.0
'''



token_X_100 = len(sentences)                   # sentences: 33793
token_X_010 = 30                               # sentences length: 30
token_X_001 = W.shape[1]                       # encode word length: 100
token_Y_100 = len(sentences)                   # sentences: 33793
token_Y_010 = 30                               # sentences length: 30
token_Y_001 = 19                               # PHI level: 18+1


text_sentences = []

for sentence in sentences:
    tokens = [" "*char_X_010] * 30
    temp_tokens = sentence.split()
    for i in range(0, min(len(temp_tokens),30)):
        tokens[i] = temp_tokens[i].ljust(char_X_010)[0:char_X_010]
    text_sentences.append(tokens)


X = np.zeros((token_X_100, token_X_010, token_X_001), dtype=np.float64) # X.shape => (33793, 30, 100)
y = np.zeros((token_Y_100, token_Y_010, token_Y_001), dtype=np.float64) # X.shape => (33793, 30, 19)

for i in range(0, len(text_sentences)):
    char_x_01 = np.zeros((1, token_X_010, token_X_001), dtype=np.float64)  # char_x_01.shape => (1,30,100)
    sentence = text_sentences[i]
    for j in range(0, len(sentence)):
        char_x_001 = np.zeros((1, char_X_010, char_X_001), dtype=np.bool)  # char_x_001.shape => (1,32,37)
        token = sentence[j]
        for k in range(0, len(token)):
            char_x_001[0, k, char_indices[token[k]]] = 1

        if token.strip() in vocab.keys():
            map_GloVe = W[vocab[token.strip()][0]].reshape(1, 100)
            map_LSTM = model.predict([char_x_001, char_x_001], verbose=0)
            encode = np.mean([map_GloVe, map_LSTM], axis=0)  # encode.shape => (1,100)
        else:
            encode = model.predict([char_x_001, char_x_001], verbose=0)  # encode.shape => (1,100)
        char_x_01[0][j] = encode
    X[i] = char_x_01[0]





print "end"