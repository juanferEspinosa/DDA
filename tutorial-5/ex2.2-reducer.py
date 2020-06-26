#!/usr/bin/python
import sys
import numpy as np

airport_delays = {}
avg_delay_airports = []
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

for i, airport in enumerate(airport_delays.keys()):
    avg_delay = sum(airport_delays[airport])*1.0 / len(airport_delays[airport])
    avg_delay_airports.append([airport, avg_delay])
avg_delay_airports = np.array(avg_delay_airports)   
idx = (avg_delay_airports[:,1]).argsort()[:10]
top_10 = avg_delay_airports[idx]
print('top_10',top_10)