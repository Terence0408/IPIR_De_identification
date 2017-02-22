#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path
import sys

from gensim.corpora import WikiCorpus

if __name__ == '__main__':
    path = "/home/terence/pycharm_use/IPIR_De_identification/"
    filen = path + "3_01_use_wiki_giga.py"
    inp = path + "1_data/outside/"+".xml.bz2"
    outp = path + "1_data/outside/"+"wiki.txt"

    program = os.path.basename(filen)
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join([filen, inp, outp]))

    space = " "
    i = 0

    output = open(outp, 'w')
    wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        output.write(space.join(text) + "\n")
        i = i + 1
        if (i % 10000 == 0):
            logger.info("Saved " + str(i) + " articles")

    output.close()
    logger.info("Finished Saved " + str(i) + " articles")