#!/usr/bin/env python
import sys

for line in sys.stdin:
    line = line.split()

    for word in line:
        print('%s\t%s' % (word, 1))


