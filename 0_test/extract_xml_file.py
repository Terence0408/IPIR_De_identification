#!/usr/bin/python
# -*- coding: utf-8 -*-

import psycopg2
import glob, os
import re
import xml.etree.ElementTree as ET

path = "/home/terence/pycharm_use/RNN/"

get_conn = psycopg2.connect(dbname='Demo DB',user='postgres', host='localhost', password='postgres')
get_conn.autocommit = True
get_cur  = get_conn.cursor()


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


os.chdir(path +"1_data/i2b2/training-PHI-Gold-Set1/")
subject = glob.glob("*.xml")
os.chdir(path +"1_data/i2b2/training-PHI-Gold-Set2/")
subject += glob.glob("*.xml")
train_len=len(subject)
os.chdir(path +"1_data/i2b2/testing-PHI-Gold-fixed/")
subject += glob.glob("*.xml")

for i in range(0, len(subject)):
    if i<train_len:
        train = 1
    else:
        train = 0
    (subject_id, order_id) = re.findall(r"[\w']+", subject[i])[0:2]

    tree = ET.parse(path +"1_data/i2b2/training-PHI-Gold-Set1/"+subject[i])
    root = tree.getroot()




root.tag, root[0].tag, root[1].tag
# ('deIdi2b2', 'TEXT', 'TAGS')

root[0].text




root[1][0].attrib