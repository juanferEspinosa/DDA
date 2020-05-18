# Tutorial 2 Exercise 1
# Juan Espinosa Reinoso
# 303158

from mpi4py import MPI
import numpy as np 
import pandas as pd
import os

comm = MPI.COMM_WORLD
worker = comm.Get_rank()
num_workers = comm.Get_size()
name = MPI.Get_processor_name()

def import_data():
    path = "/Users/juanfer/Documents/Maestria/SemesterIII/Lab-DDA/tutorial-2/20_newsgroups"
    folders = os.listdir(path)
    News = []
    Group = []
    for folder in folders:
        for file in os.listdir(os.path.join(path,folder)):
            with open(os.path.join(path,folder,file),encoding='latin-1') as opened_file:
                New = opened_file.read().lower().split()
                News.append(New)
                Group.append(folder)
    return News, Group


stop = ['a','able','about','across','after','all','almost','also','am','among','an','and','any','are','as','at','be','because','been','but',
        'by','can','cannot','could','dear','did','do','does','either','else','ever','every','for','from','get','got','had','has','have','he','her',
        'hers','him','his','how','however','i','if','in','into','is','it','its','just','least','let','like','likely','may','me','might','most','must',
        'my','neither','no','nor','not','of','off','often','on','only','or','other','our','own','rather','said','say','says','she','should','since',
        'so','some','than','that','the','their','them','then','there','these','they','this','tis','to','too','twas','us','wants','was','we','were',
        'what','when','where','which','while','who','whom','why','will','with','would','yet','you','your']



def cleaning(data):
    lista = []
    for i in range(len(data)):
        News_clean =  [j for j in data[i] if j not in stop and j.isalpha()]
        lista.append(News_clean)
    return lista


global_time = MPI.Wtime()
News, group = import_data()
p_workers = round(len(News)/(num_workers-1))
# Initializing the parallelization process. 

if worker == 0:
    for i in range(1, num_workers):
        News1 = News[((i-1)*p_workers):(p_workers*i)]
        comm.send(News1, dest=i)
    
else:
    data = comm.recv()
    News = cleaning(data)
    comm.send(News, dest=0)

if worker == 0:

    for i in range(1, num_workers):
        news_list = []
        output  = comm.recv()
        News_list = news_list + output
    for count, item in enumerate(News_list, 1):
        with open(os.path.join('preprocess',f'name{count}.txt'), 'w') as f:
            f.write("%s\n" % ','.join(item))

print('Final Time:',MPI.Wtime() - global_time)