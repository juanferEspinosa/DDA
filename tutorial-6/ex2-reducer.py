#!/usr/bin/env python
import numpy as np
import os
import sys

words_cleaned = {}
for i, line in enumerate(sys.stdin):
    line = line.split()
    (word, count) = line
    if count not in words_cleaned:
        words_cleaned[count] = []
        words_cleaned[count].append(word)
    else:
        words_cleaned[count].append(word)
result = words_cleaned.pop('0')
print(words_cleaned)



