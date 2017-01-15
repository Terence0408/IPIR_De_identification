#!/usr/bin/python
# -*- coding: utf-8 -*-

# Goal: Build label prediction layer and label sequence optimization layer to predict PHI type.
# Step: 1. Build bidirectional lstm layer.
#             Input: token(100 dimension) sequence of a sentence.
#             Output: 100 dimension.
#       2. Add dense layer
#             Input: 100 dimension(lstm layer result)
#             Output: 24 prob. dimension
#       3. Add label sequence optimization layer
#             Input: 24 prob. dimension(dense layer result)
#             Output: 24 PHI label(binary vector)
#       Train step: x: token(100 dimension) sequence of a sentence.
#                   y: 24 PHI label(18 PHI-type and 1 non PHI)


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

vocab = pickle.load(open(path+"model/GloVe_vocab.pk", "rb" ))
W = pickle.load(open(path+"model/GloVe_W.pk", "rb" ))
model = load_model(path+"model/biLSTM_char.pk")

get_cur.execute("Select row_id, subject_id, order_id, sentence_id, sentence, labels from sentence_text where train = 1;")# where subject_id= 253 and order_id =3;")
table = get_cur.fetchall()
sentences = []
for row in table:
    sentences.append([row[0],row[1],row[2],row[3],row[4].split(),row[5].split()])


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
chara_x_010 = len(max(vocab.keys(), key=len))  # max word length: 54
chara_x_001 = len(chars)                       # chars: 37
chara_y_10  = 1                                # words: 1
chara_y_01  = W.shape[1]                       # encode word length: 100

token_x_100 = len(sentences)                   # sentences: 33700
token_x_010 = 30                               # sentences length: 30
token_x_001 = W.shape[1]                       # encode word length: 100
token_y_100 = len(sentences)                   # sentences: 33700
token_y_010 = 30                               # sentences length: 30
token_y_001 = 24                               # PHI level: 23+1


# load model and test.
'''
if "having" in vocab.keys():
    map_GloVe = W[vocab['having'][0]]

text = 'having'.ljust(len_X_010)
x = np.zeros((1, len_X_010, len_X_001), dtype=np.bool)

for j in range(0, len(text)):
    x[0, j, char_indices[text[j]]] = 1
map_LSTM = model.predict([x,x], verbose=0)

print 'Load model and test.'
'''
# Define sentence length 95% sentence less then 30 words.
'''
lens=[]
for i in sentences:
    lens.append(len(i[4]))
arlens=np.asarray(lens)
print "Define sentence length 91.5% sentence less then "+ str(np.percentile(arlens, 91.5)) +" words." # 30.0
'''

# Adjust tokens into fixed size, adjust sentences and labels into fixed size.
adjusts = []
for i in range(0, len(sentences)):
    sentence = sentences[i]

    tokens = [" "*chara_x_010] * 30
    for j in range(0,min(len(sentences[i][4]), token_x_010)):
        tokens[j] = sentences[i][4][j][0:min(len(sentences[i][4][j]),chara_x_010)].ljust(chara_x_010)

    labels = ['NONE'] * token_y_010
    for j in range(0,min(len(sentences[i][5]), token_x_010)):
        labels[j] = sentences[i][5][j]

    adjusts.append([sentence[0], sentence[1], sentence[2], sentence[3], tokens, labels])



# Transform sentences and labels into numpy array.
token_x = np.zeros((token_x_100, token_x_010, token_x_001), dtype=np.float64) # token_x.shape => (33700, 30, 100)
token_y = np.zeros((token_y_100, token_y_010, token_y_001), dtype=np.bool)    # token_y.shape => (33700, 30, 24)
for i in range(0, len(adjusts)):
    sentence = adjusts[i]
    sentence_x = np.zeros((token_x_010,  token_x_001), dtype=np.float64) # sentence_x.shape: (30,100)
    sentence_y = np.zeros((token_y_010,  token_y_001), dtype=np.bool)    # sentence_x.shape: (30, 19)
    for j in range(0, token_x_010): # range(0,30)

        # Transform sentences into numpy array.

        chars = sentence[4][j]
        # encoding token by biLSTM.
        chara_x = np.zeros((1, chara_x_010, chara_x_001), dtype=np.bool)  # chara_x_001.shape => (1, 54, 37)
        for k in range(0, chara_x_010): # range(0,54)
            chara_x[0, k, char_indices[chars[k]]] = 1
        encode_biLSTM = model.predict([chara_x, chara_x], verbose=0)

        # encoding token by GloVe(if have), then concentrate them. If token not in GloVe vocab., only use biLSTM
        if chars.strip() in vocab.keys():
            encode_GloVe = W[vocab[chars.strip()][0]].reshape(1, 100)
            encode = np.mean([encode_biLSTM, encode_GloVe], axis=0)  # encode.shape => (1,100)
        else:
            encode = encode_biLSTM
        sentence_x[j] = encode


        # Transform labels into numpy array.
        lable = sentence[5][j]
        lable_y = np.zeros((1, token_y_001), dtype=np.bool) # lable_y.shape => (1, 24)
        lable_y[0, labe_indices[lable]] = 1
        sentence_y[j] = lable_y

    token_x[i] = sentence_x
    token_y[i] = sentence_y

pickle.dump(token_x,  open(path+"model/token_x.pk", "wb"))
pickle.dump(token_y,  open(path+"model/token_y.pk", "wb"))


# build the model: a bidirectional LSTM

token_x = pickle.load(open(path+"model/token_x.pk", "rb" ))
token_y = pickle.load(open(path+"model/token_y.pk", "rb" ))

print('Build model...')
left = Sequential()
left.add(LSTM(100, input_shape=(token_x_010, token_x_001), activation='tanh',
              inner_activation='sigmoid', dropout_W=0.5, dropout_U=0.5,
              return_sequences=True))

right = Sequential()
right.add(LSTM(100, input_shape=(token_x_010, token_x_001), activation='tanh',
               inner_activation='sigmoid', dropout_W=0.5, dropout_U=0.5, go_backwards=True,
               return_sequences=True))


label_model = Sequential()
label_model.add(Merge([left, right], mode='sum'))
label_model.add(TimeDistributed(Dense((24), activation='sigmoid')))
#model.add(MaxoutDense(pool_length = 1))

label_model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

label_model.fit([token_x,token_x], token_y,
          batch_size=128,
          nb_epoch=10)
label_model.save(path+"model/biLSTM_label.pk")

pred = label_model.predict([token_x[0:1],token_x[0:1]], verbose=0)
pred_labels=[]
for j in range(0,token_y_010):
    pred_labels.append(indices_labe[np.argmax(pred[0][j])])

print "end"