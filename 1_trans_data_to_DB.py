#!/usr/bin/python
# -*- coding: utf-8 -*-

import psycopg2
import glob, os
import re
import xml.etree.ElementTree as ET

path = "/home/terence/pycharm_use/IPIR_De_identification/1_data/"
get_conn = psycopg2.connect(dbname='IPIR_De_identification',user='postgres', host='localhost', password='postgres')


get_conn.autocommit = True
get_cur  = get_conn.cursor()

get_cur.execute("DROP TABLE IF EXISTS Record_text;;"
                "DROP TABLE IF EXISTS Record_PHI;;")
get_cur.execute("Create table Record_text ("
                "  row_id serial primary key, "
                "  subject_id integer,"
                "  order_id integer,"
                "  content text,"
                "  train integer);"
                "Create table Record_PHI ("
                "  row_id serial primary key,"
                "  subject_id integer,"
                "  order_id integer,"
                "  id integer,"
                "  type text,"
                "  text text,"
                "  text_start integer,"
                "  text_end integer,"
                "  comment text,"
                "  train integer);")


os.chdir(path +"i2b2/training-PHI-Gold-Set1/")
subject = glob.glob("*.xml")
subject_len = [len(subject)]

os.chdir(path +"i2b2/training-PHI-Gold-Set2/")
subject += glob.glob("*.xml")
subject_len.append(len(subject))

os.chdir(path +"i2b2/testing-PHI-Gold-fixed/")
subject += glob.glob("*.xml")
subject_len.append(len(subject))

for i in range(0, subject_len[2]):

    (subject_id, order_id) = re.findall(r"[\w']+", subject[i])[0:2]

    if i<subject_len[0]:
        tree = ET.parse(path + "i2b2/training-PHI-Gold-Set1/" + subject[i])
        train = '1'
    elif i<subject_len[1]:
        tree = ET.parse(path + "i2b2/training-PHI-Gold-Set2/" + subject[i])
        train = '1'
    else:
        tree = ET.parse(path + "i2b2/testing-PHI-Gold-fixed/" + subject[i])
        train = '0'

    root = tree.getroot()

    get_cur.execute("Insert into Record_text (subject_id, order_id, content, train) "
                    "values "
                    "("+subject_id+","+order_id+",'"+root[0].text.replace("'", "''")+"', "+train+");")
    for j in range(0,len(root[1])):

        get_cur.execute("Insert into Record_PHI (subject_id, order_id, id, type, text, text_start, text_end, comment, train) "
                        "values "
                        "(" + subject_id + "," + order_id + ", "
                        "" + root[1][j].attrib['id'].replace("P", "") + ", "
                        "'" + root[1][j].attrib['TYPE'] + "', "
                        "'" + root[1][j].attrib['text'].replace("'", "''") + "', "
                        "'" + root[1][j].attrib['start'] + "', "
                        "'" + root[1][j].attrib['end'] + "', "
                        "'" + root[1][j].attrib['comment'].replace("'", "''") + "', " + train + ");")



print "end"