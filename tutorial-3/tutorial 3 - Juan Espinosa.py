from mpi4py import MPI
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import random 
from collections import defaultdict

comm = MPI.COMM_WORLD
worker = comm.Get_rank()
num_workers = comm.Get_size()
name = MPI.Get_processor_name()

# function that retrieves the distance matrix between workers
def euclidean_distance(arrays, centers):
    distance_matrix = pairwise_distances(arrays,centers)
    return distance_matrix

# function that retrieves the new centers for each worker
def new_cluster(distance, arrays):

    center_idx = np.argmin(distance, axis=0)
    argmin_matrix = np.argmin(distance, axis=1)
    clusters_new = []
    for i in range(len(center_idx)):
        doc_index = np.where(i == argmin_matrix)
        documents_center = arrays[doc_index]
        mean = np.sum(documents_center, axis=0) / documents_center.shape[0]
        clusters_new.append(mean)
    clusters_new = np.array(clusters_new)
    return clusters_new

initial_time = MPI.Wtime()

if worker == 0:
    # Number of clusters
    K = 6
    np.random.seed(35)
    categories = ['alt.atheism','alt.atheism','comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware',
                'comp.sys.mac.hardware','comp.windows.x'] 
    Newsgroup = fetch_20newsgroups(subset='train', categories = categories)
    vectorizer = CountVectorizer()
    vector = vectorizer.fit_transform(Newsgroup.data)
    vectorizer2 = TfidfTransformer()
    news = vectorizer2.fit_transform(vector).toarray()
    # Splittin the data for each worker
    p_workers = round(news.shape[0]/(num_workers))
    data = [news[(i*p_workers):(p_workers*(i+1))] for i in range(num_workers)]  
    idx = random.sample(range(0, news.shape[0]), k=K)
    # random initialization for the first centers
    centers = news[idx]

else:
    data = None
    centers = None

# Scattering the data in equal parts to each worker
data = comm.scatter(data, root=0)

# Broadcasting the centers to all the workers
centers = comm.bcast(centers, root=0)
max_iterations = 100
iteration = 0

for _ in range(max_iterations):

    #  Measuring the distance and new clusters in each worker
    distance = euclidean_distance(data, centers)
    new_clusters = new_cluster(distance, data)

    comm.Barrier()

    new_centers = comm.reduce(new_clusters, op=MPI.SUM, root=0)

    if worker == 0:
        new_centers = np.array(new_centers) / num_workers
        print()
        if ((1/centers.shape[0])*(np.sum(new_centers - centers)**2)) < 0.01:
            break
        #print('Final centers:', new_centers)
        centers = new_clusters
        iteration = iteration + 1
        #print('iteration:', iteration)
    centers = comm.bcast(centers, root=0)

if worker == 0:
    print('Final centers:', centers)

print('Final Time:',MPI.Wtime() - initial_time)