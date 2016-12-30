#!/usr/bin/python
# -*- coding: utf-8 -*-

# Goal: Train i2b2 dataset by GloVe.
# Parameters: 1. window size: 10
#             2. vocabulary count: 5
#             3. iterations: 15
#             4. Output dimension: 100

import logging
import cPickle as pickle
import psycopg2
import numpy as np
np.random.seed(19870712)  # for reproducibility
path = "/home/terence/pycharm_use/IPIR_De_identification/1_data/"
get_conn = psycopg2.connect(dbname='IPIR_De_identification',user='postgres', host='localhost', password='postgres')


import nltk
import re
from GloVe import evaluate
from GloVe import glove


get_conn.autocommit = True
get_cur  = get_conn.cursor()

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

get_cur.execute("Select row_id, subject_id, order_id, content from record_text;")# where subject_id= 253 and order_id =3;")
table = get_cur.fetchall()



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

print len(sentences) #56328


glove.logger.setLevel(logging.ERROR)
vocab = glove.build_vocab(sentences)
cooccur = glove.build_cooccur(vocab, sentences, window_size=10, min_count=5)
id2word = evaluate.make_id2word(vocab)

W = glove.train_glove(vocab, cooccur, vector_size=100, iterations=500)


# Merge and normalize word vectors
W = evaluate.merge_main_context(W)

#pickle W and vocab
pickle.dump(vocab,  open(path+"model/GloVe_vocab.pk", "wb" ))
pickle.dump(W, open(path+"model/GloVe_W.pk", "wb" ))

#vocab = pickle.load(open(path+"model/GloVe_vocab.pk", "rb" ))
#W = pickle.load(open(path+"model/GloVe_W.pk", "rb" ))

#print W[vocab[id2word[3]][0]]
#if "having" in vocab.keys():
#    print W[vocab['having'][0]]



print "end"