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

get_cur.execute("Alter table record_text "
                "Drop column if exists clean_content; "
                "Alter table record_text "
                "Add column clean_content text; "
                "Alter table record_phi "
                "Drop column if exists clean_text; "
                "Alter table record_phi "
                "Drop column if exists clean_position; "
                "Alter table record_phi "
                "Add column clean_text text; "
                "Alter table record_phi "
                "Add column clean_position integer; ")



'''
# Clean content.
get_cur.execute("Select row_id, subject_id, order_id, content from record_text")
table = get_cur.fetchall()

for row in table:
    half_clean = re.sub("[^0-9a-zA-Z]"," ",row[3])
    ter_query  = "Select row_id, subject_id, order_id, text, id "
    ter_query += "from record_phi "
    ter_query += "where subject_id = "+str(row[1]) +" and order_id = "+str(row[2])+" "
    ter_query += "order by id ;"
    get_cur.execute(ter_query)
    phi_table = get_cur.fetchall()

    for phi_row in phi_table:
        phi = re.sub(" ", "_", phi_row[3])
        phi_clean_half = re.sub("[^0-9a-zA-Z]", " ", phi)
        phi_clean = re.sub("[^0-9a-zA-Z]", "", phi)
        half_clean = re.sub(phi_clean_half, phi, half_clean, 1)

    half_clean = " ".join(half_clean.split())

    for phi_row in phi_table:
        phi = re.sub(" ", "_", phi_row[3])
        phi_clean_half = re.sub("[^0-9a-zA-Z]", " ", phi)
        phi_clean = re.sub("[^0-9a-zA-Z]", "", phi)
        location = half_clean[0:half_clean.find(phi)].count(' ')
        half_clean = re.sub(phi, phi_clean, half_clean, 1)

        ter_query  = "Update record_phi set (clean_text, clean_position) = ('"+phi_clean.lower()+"',"+str(location)+") "
        ter_query += "Where row_id = "+str(phi_row[0])+";"
        get_cur.execute(ter_query)

    ter_query = "Update record_text set (clean_content) = ('" + half_clean.lower() + "') Where row_id = "+str(row[0])+";"
    get_cur.execute(ter_query)
'''


# Clean sentence.
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

get_cur.execute("Select row_id, subject_id, order_id, content, train from record_text;")# where subject_id= 253 and order_id =1;")
table = get_cur.fetchall()

for row in table:
    ter_query = "Select row_id, subject_id, order_id, text, id ,type, clean_text "
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
    sentences = tokenizer.tokenize(content)


    for sentence_id in range(0, len(sentences)):
        half_clean = re.sub("[^0-9a-zA-Z]", " ", sentences[sentence_id])

        for phi_row in phi_table:
            phi = re.sub(" ", "_", phi_row[3])
            phi_clean_half = re.sub("[^0-9a-zA-Z]", " ", phi)
            phi_clean = re.sub("[^0-9a-zA-Z]", "", phi)
            half_clean = re.sub(phi_clean_half, phi, half_clean)

        half_clean = " ".join(half_clean.split())

        for phi_row in phi_table:
            phi = re.sub(" ", "_", phi_row[3])
            phi_clean_half = re.sub("[^0-9a-zA-Z]", " ", phi)
            phi_clean = re.sub("[^0-9a-zA-Z]", "", phi)
            half_clean = re.sub(phi, phi_clean, half_clean, 1)
        clean = " ".join(half_clean.lower().split())

        ter_query  = "Insert into sentence_text (subject_id, order_id, sentence_id, sentence, train) "
        ter_query += "Values ("+str(row[1])+", "+str(row[2])+", "+str(sentence_id)+", '"+clean+"', "+str(row[4])+");"
        get_cur.execute(ter_query)

        for j in range(0, len(phi_table)):
            locations = [i for i, x in enumerate(clean.split()) if x == phi_table[j][6]]
            if len(locations)>0:
                ter_query  = "Insert into sentence_phi (subject_id, order_id, sentence_id, phi_id, text, type, location, train) "
                ter_query += "Values ("+str(row[1])+", "+str(row[2])+", "+str(sentence_id)+", "
                ter_query += "        "+str(phi_table[j][4])+", '"+str(phi_table[j][6])+"', " # insert phi_id, text
                ter_query += "        '"+str(phi_table[j][5])+"', " + str(locations[0])+", "+str(row[4])+");"
                get_cur.execute(ter_query)


print "end."








