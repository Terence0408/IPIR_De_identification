#!/usr/bin/python
# -*- coding: utf-8 -*-

# Goal: Train i2b2 dataset by GloVe.
# Parameters: 1. window size: 10
#             2. vocabulary count: 5
#             3. iterations: 15
#             4. Output dimension: 100


import psycopg2
from bs4 import BeautifulSoup
get_conn = psycopg2.connect(dbname='IPIR_De_identification',user='postgres', host='localhost', password='postgres')
get_conn.autocommit = True
get_cur  = get_conn.cursor()


get_cur.execute("DROP TABLE IF EXISTS Record_text;;"
                "DROP TABLE IF EXISTS Record_PHI;;")

