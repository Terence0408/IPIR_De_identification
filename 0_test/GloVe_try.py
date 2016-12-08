#!/usr/bin/python
# -*- coding: utf-8 -*-





import itertools
from gensim.models.word2vec import Text8Corpus
from glove import Corpus, Glove

path = "/home/terence/pycharm_use/RNN/0_test/data/"
sentences = list(itertools.islice(Text8Corpus(path+'text8'),None))

sentences[0] # [u'anarchism', u'originated',..., u'or', u'emotional']


corpus = Corpus()
corpus.fit(sentences, window=10)
glove = Glove(no_components=100)
glove.fit(corpus.matrix, epochs=15, no_threads=5)
glove.add_dictionary(corpus.dictionary)
glove.most_similar('man')

print 'end'