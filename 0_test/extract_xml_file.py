#!/usr/bin/python
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET

path = "/home/terence/pycharm_use/RNN/"

tree = ET.parse(path +"1_data/i2b2/training-PHI-Gold-Set1/220-01.xml")
root = tree.getroot()

root.tag, root[0].tag, root[1].tag
# ('deIdi2b2', 'TEXT', 'TAGS')

