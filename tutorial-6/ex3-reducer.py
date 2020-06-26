#!/usr/bin/env python

import sys
import numpy as np
from collections import Counter

lista = []
TF = []
TFIDF = []
documents = {}
for line in sys.stdin:
    file_word, count = line.split('\t')
    document, words = file_word.split(',')
    lista.append((document,words))

for i,e in lista:
    if i in documents.keys():
        documents[i].append(e)
    else:
        documents[i] = []
        documents[i].append(e)

number_documents = int(len(documents.keys()))

for i in documents.values():
    frequency = Counter(i)
    unique_words = int(len(list(set(i))))
    partial_tf = {k: (float(v) / float(unique_words)) for k,v in frequency.items()}
    TF.append(partial_tf)

documents_token = Counter(i for j in TF for i in j)
IDF = {k: (np.log(float(number_documents) / float(v))) for k,v in documents_token.items()}


for i in TF:
    result_dict = {}
    for key,value in i.items():
        for key1,value1 in IDF.items():
            result_dict["{}".format(key)] = '{:.3f}'.format(value * value1)
    TFIDF.append(result_dict)
print(TFIDF)







