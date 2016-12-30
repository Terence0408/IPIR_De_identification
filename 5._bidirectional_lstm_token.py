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


from keras.models import Sequential
from keras.layers import Dense, Activation, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file


path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
        y[i, t, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars)),activation='tanh', inner_activation='sigmoid', dropout_W=0.2, dropout_U=0.2, return_sequences=True))
model.add(TimeDistributed(Dense(len(chars),activation='softmax')))




model.compile(loss='categorical_crossentropy', optimizer='sgd')

model.fit(X, y, batch_size=128, nb_epoch=1)

for i in range(400):
    x = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.

    preds = model.predict(x, verbose=0)[0]