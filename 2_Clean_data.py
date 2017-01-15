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
get_conn.autocommit = True
get_cur  = get_conn.cursor()


# Clean content.
'''
get_cur.execute("Select row_id, subject_id, order_id, content from record_text;")# where subject_id= 109 and order_id =1;")
table = get_cur.fetchall()

for row in table:
    half_clean = re.sub("\n"," ",re.sub("\+","",re.sub("\)", "", re.sub("\(", "", row[3]))))
    ter_query  = "Select row_id, subject_id, order_id, text, id "
    ter_query += "from record_phi "
    ter_query += "where subject_id = "+str(row[1]) +" and order_id = "+str(row[2])+" "
    ter_query += "order by id ;"
    get_cur.execute(ter_query)
    phi_table = get_cur.fetchall()

    for phi_row in phi_table:
        phi = re.sub("\n"," ",re.sub("\+","",re.sub("\)", "",re.sub("\(", "", phi_row[3]))))
        phi_clean = re.sub("[^0-9a-zA-Z]", "", phi)
        half_clean = re.sub(phi, phi_clean, half_clean, 1)

        ter_query  = "Update record_phi set (clean_text) = ('"+phi_clean.lower()+"') "
        ter_query += "Where row_id = "+str(phi_row[0])+";"
        get_cur.execute(ter_query)


    half_clean = " ".join(re.sub("[^0-9a-zA-Z]", " ", half_clean).lower().split())

    ter_query = "Update record_text set (clean_content) = ('" + half_clean + "') Where row_id = "+str(row[0])+";"
    get_cur.execute(ter_query)
print 'Clean content.'
'''

# Find phi location in content.
'''
get_cur.execute("Select row_id, subject_id, order_id, clean_content, train from record_text;")# where subject_id= 388 and order_id =1;")
table = get_cur.fetchall()
for row in table:
    ter_query = "Select row_id, subject_id, order_id, text, id ,type, clean_text "
    ter_query += "from record_phi "
    ter_query += "where subject_id = " + str(row[1]) + " and order_id = " + str(row[2]) + " "
    ter_query += "order by id ;"
    get_cur.execute(ter_query)
    phi_table = get_cur.fetchall()
    position = 0
    for phi_row in phi_table:
        position = row[3][position:].find(phi_row[6])+position
        location = row[3][0:position].count(' ')

        ter_query = "Update record_phi set (clean_position) = (" + str(location) + ") Where row_id = "+str(phi_row[0])+";"
        get_cur.execute(ter_query)
print 'Find phi location in content.'
'''

# Clean sentence.
'''
get_cur.execute("DROP TABLE IF EXISTS sentence_text;")
get_cur.execute("Create table sentence_text ("
                "  row_id serial primary key, "
                "  subject_id integer,"
                "  order_id integer,"
                "  sentence_id integer,"
                "  sentence text,"
                "  labels text,"
                "  train integer);")
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

get_cur.execute("Select row_id, subject_id, order_id, content, train from record_text;")# where subject_id= 287 and order_id =2;")
table = get_cur.fetchall()

for row in table:

    ter_query = "Select row_id, subject_id, order_id, text, id ,type, clean_text "
    ter_query += "from record_phi "
    ter_query += "where subject_id = " + str(row[1]) + " and order_id = " + str(row[2]) + " "
    ter_query += "order by id ;"
    get_cur.execute(ter_query)
    phi_table = get_cur.fetchall()

    content = re.sub("\n"," ",re.sub("\+","",re.sub("\)", "",re.sub("\(", "", row[3]))))

    for phi_row in phi_table:
        content = re.sub(phi_row[3], phi_row[6], content,1)

    dirty_sentences = tokenizer.tokenize(content)
    sentences = []
    for sentence in dirty_sentences:
        half_clean = re.sub("[^0-9a-zA-Z]", " ", sentence)
        clean = " ".join(half_clean.lower().split())
        if len(clean)>0:
            sentences.append(clean)

    for sentence_id in range(0, len(sentences)):
        ter_query  = "Insert into sentence_text (subject_id, order_id, sentence_id, sentence, train) "
        ter_query += "Values ("+str(row[1])+", "+str(row[2])+", "+str(sentence_id)+", '"+sentences[sentence_id]+"', "+str(row[4])+");"
        get_cur.execute(ter_query)
print 'Clean sentence.'
'''

# Find phi location in sentence.
'''
get_cur.execute("Select row_id, subject_id, order_id, clean_content from record_text;")# where subject_id= 287 and order_id =2;")
table = get_cur.fetchall()

lengths =[0]
length = 0
for row in table:

    ter_query = "Select row_id, subject_id, order_id, sentence_id, sentence "
    ter_query += "from sentence_text where subject_id= "+str(row[1])+" and order_id ="+str(row[2])+" "
    ter_query += "order by sentence_id;"
    get_cur.execute(ter_query)
    sentence_table = get_cur.fetchall()
    lengths = [0]
    length = 0
    for sentence_row in sentence_table:
        length += len(sentence_row[4].split())
        lengths.append(length)

    ter_query = "Select row_id, subject_id, order_id, clean_position "
    ter_query += "from record_phi "
    ter_query += "where subject_id = " + str(row[1]) + " and order_id = " + str(row[2]) + " "
    ter_query += "order by id ;"
    get_cur.execute(ter_query)
    phi_table = get_cur.fetchall()

    for phi_row in phi_table:
        for sentence_id in range(0, len(lengths)-1):
            if phi_row[3] in range(lengths[sentence_id], lengths[sentence_id+1]):
                sentence_position = phi_row[3]-lengths[sentence_id]

                ter_query = "Update record_phi set (sentence_id, sentence_position) = (" + str(sentence_id) + ", "+str(sentence_position)+") "
                ter_query += "Where row_id = " + str(phi_row[0]) + ";"
                get_cur.execute(ter_query)
print 'Find phi location in sentence.'
'''

# Add phi labels in sentence.
'''
get_cur.execute("Select row_id, subject_id, order_id, sentence_id, sentence from sentence_text;")# where subject_id= 287 and order_id =2;")
table = get_cur.fetchall()
for row in table:
    ter_query = "Select row_id, subject_id, order_id, sentence_id ,type, sentence_position "
    ter_query += "from record_phi "
    ter_query += "where subject_id = " + str(row[1]) + " and order_id = " + str(row[2]) + " and "
    ter_query += "      sentence_id = " +str(row[3])+" "
    ter_query += "order by sentence_position ;"
    get_cur.execute(ter_query)
    phi_table = get_cur.fetchall()
    labels = ['NONE']*len(row[4].split())
    for phi_row in phi_table:
        labels[phi_row[5]] = phi_row[4]
    ter_query = "Update sentence_text set (labels) = ('" +" ".join(labels)+ "') "
    ter_query += "Where row_id = " + str(row[0]) + ";"
    get_cur.execute(ter_query)
print 'Add phi labels in sentence.'
'''


print "end."