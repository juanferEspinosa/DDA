#!/usr/bin/python
import sys


airport_delays = {}

for i, line in enumerate(sys.stdin):
    if i == 0:
        continue
    line = line.strip()
    (airport,delay_dep) = line.split('\t')
    
    if airport in airport_delays:
        airport_delays[airport].append(float(delay_dep))
    else:
        airport_delays[airport] = []
        airport_delays[airport].append(float(delay_dep))

for airport in airport_delays.keys():
    max_delay = max(airport_delays[airport])
    avg_delay = sum(airport_delays[airport])*1.0 / len(airport_delays[airport])
    min_delay = min(airport_delays[airport])
    print('Maximum',(airport, max_delay))
    print('Average', (airport, avg_delay))
    print('Minimum', (airport, min_delay))