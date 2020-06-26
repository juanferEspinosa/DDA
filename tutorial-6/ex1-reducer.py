#!/usr/bin/env python
import sys

word_count = {}
counting = 0
for line in sys.stdin:
    (words, count) = line.split('\t')
    if words in word_count:
         word_count[words] += 1
    else:
        word_count[words] = 1
print(word_count)
    
    

