# Tutorial 2 Exercise 4
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
    path1 = "/Users/juanfer/Documents/Maestria/SemesterIII/Lab-DDA/tutorial-2/prueba32"
    folders = os.listdir(path1)
    News = []
    for folder in folders:
        with open(os.path.join(path1,folder),encoding='latin-1') as opened_file:
            New = opened_file.read().split(',')
            News.append(New)

    return News


def TFIDF(data):
    chunk = []
    word_folder = dict()
    for i in range(len(data)):
        word = defaultdict(int)
        unique_tf = np.unique(data[i])
        counter = len(np.unique(unique_tf))
        for i in data[i]:
            word[i] +=1
        dictionary = {k: v / counter for k, v in word.items()}
        chunk.append(dictionary)
        for unique in unique_tf:
            if unique in word_folder:
                word_folder[unique] += 1
            else:
                word_folder[unique] = 1
    batch = {k: (len(data) / v) for k, v in word_folder.items()}

    return chunk, batch


initial_time = MPI.Wtime()
News = import_data()
p_workers = round(len(News)/(num_workers-1))

# INITIALIZATION MPI
if worker == 0:
    News2 = News[0:p_workers]
    for i in range(1, num_workers):
        News1 = News[(i*p_workers):(p_workers*(i+1))]
        comm.send(News1, dest=i)
    TF1, IDF1 = TFIDF(News2)
    

else:
    data = comm.recv()
    TF, IDF = TFIDF(data)
    output = (TF, IDF)
    comm.send(output, dest=0)


final_idf = dict()
final = []
dictTFIDF = dict()


if worker == 0:
    global_dict = None
    for i in range(1, num_workers):
        chunka = []
        TF2, IDF2 = comm.recv()
        TF1 = TF1 + TF2
        IDF1 = (Counter(IDF2) + Counter(IDF1))
    final = TF1 + final
    final_idf = {k: np.log(v / (num_workers-1)) for k, v in IDF1.items()}
    for i in range(len(final)):
        n = final[i]
        for key, value in n.items():
            if key in final_idf:
                dictTFIDF[key] = (n[key]) * (final_idf[key])
            else:
                None
    

print('Chunks:', dictTFIDF)
print('Final Time1:',MPI.Wtime() - initial_time)
