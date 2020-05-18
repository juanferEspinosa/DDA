# Tutorial 2 Exercise 3
# Juan Espinosa Reinoso
# 303158

from mpi4py import MPI
import numpy as np 
import pandas as pd
import os
from collections import defaultdict, Counter

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

def word_per_document(data):
    word_folder = dict()
    for i in range(len(data)):
        # Get unique words per document
        words = np.unique(data[i])
        # Counting the times a word has been mentioned in a document
        for word in words:
            if word in word_folder:
                word_folder[word] += 1
            else:
                word_folder[word] = 1
    batch = {k: (len(data) / v) for k, v in word_folder.items()}
    return batch

# INITIALIZATION OF THE PROCESS

initial_time = MPI.Wtime()
News = import_data()
p_workers = round(len(News)/(num_workers-1))
if worker == 0:
    News2 = News[0:p_workers]
    for i in range(1, num_workers):
        News1 = News[(i*p_workers):(p_workers*(i+1))]
        comm.send(News1, dest=i)
    output1 = word_per_document(News2)

else:
    data = comm.recv()
    output = word_per_document(data)
    comm.send(output, dest=0)


total = dict()
if worker == 0:
    global_dict = None
    for i in range(1, num_workers):
        idf = comm.recv()
        output1 = (Counter(output1) + Counter(idf))
    total = {k: np.log(v / (num_workers)) for k, v in output1.items()}


#print('Final Dictionary', total)
print('Final Time:',MPI.Wtime() - initial_time)        


# FINALIZATION