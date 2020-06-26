#!/usr/bin/python
import sys


for line in sys.stdin:
    line = line.strip()
    #line = line.splitlines()
    line = line.split(",")

    if len(line) >=3:
        airport = line[3]
        delay_dep = line[6]
        if delay_dep == "" or delay_dep == ' ':
           continue
        print('%s\t%s' % (airport, delay_dep))

