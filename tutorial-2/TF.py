# Tutorial 2 Exercise 2
# Juan Espinosa Reinoso
# 303158

from mpi4py import MPI
import numpy as np 
import pandas as pd
import os
from collections import defaultdict

comm = MPI.COMM_WORLD
worker = comm.Get_rank()
num_workers = comm.Get_size()
name = MPI.Get_processor_name()

def import_data():
    path1 = "/Users/juanfer/Documents/Maestria/SemesterIII/Lab-DDA/tutorial-2/preprocessing"
    folders = os.listdir(path1)
    News = []
    for folder in folders:
        with open(os.path.join(path1,folder),encoding='latin-1') as opened_file:
            New = opened_file.read().split(',')
            News.append(New)

    return News

def TF(data):
    chunk = []
    for i in range(len(data)):
        word = defaultdict(int)
        n = data[i]
        n = np.array(n)
        counter = len(np.unique(n))
        for i in n:
            word[i] +=1
        dictionary = {k: v / counter for k, v in word.items()}
        chunk.append(dictionary)
    return chunk


initial_time = MPI.Wtime()
News = import_data()
p_workers = round(len(News)/(num_workers-1))

if worker == 0:
    News2 = News[0:p_workers]
    for i in range(1, num_workers):
        News1 = News[(i*p_workers):(p_workers*(i+1))]
        comm.send(News1, dest =i)
    output1 = TF(News2)


else:
    data = comm.recv()
    News = TF(data)
    comm.send(News, dest=0)


final = []
if worker == 0:
    for i in range(1, num_workers):
        output = comm.recv()
        output1 = output1 + output
    final = output1 + final




print('Final Time:',MPI.Wtime() - initial_time)
