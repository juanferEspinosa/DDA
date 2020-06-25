from mpi4py import MPI
import numpy as np 
import pandas as pd
import os
from collections import defaultdict
import time
import matplotlib.pyplot as plt


# -----------  PREPROCESSING FUNCTIONS -----------

def import_data(path):
    file_merge = []
    for root, sub_dirs, files in os.walk(path):
        for name in files:
            np.random.seed(3)
            file_merge.append(name)
    return file_merge
    

def sparse_matrix(data, path):
    row_counter = 0
    y, X = [], []
    for file in data:
        # Read each document
        with open(os.path.join(path,file),'r') as opened_file:
            document = opened_file.readlines()

        for line in document:
            line = line.strip('\n')
            line = line.split()
            # from the arrays for each virus (line) in each document we extract value 0 == y
            y.append(float(line[0]))
            datapoints = defaultdict(int)

            for virus in line[1:]:
                # Split the virus input and the value of itself
                virus = virus.split(":")
                key = int(virus[0])

                if row_counter < key:
                    row_counter = key
                # Append each value to the right key
                datapoints[key] = int(virus[1])
            X.append(datapoints)
    sparse = np.zeros((len(X), (row_counter+1)))

    # Since X is a list of dictionaries, iterate through each dictionary and create the final matrix
    for idx, data in enumerate(X):
        for input in data:
            sparse[idx, input] = data[input]
    return sparse, y

# -----------  SGD/NORMALIZATION FUNCTIONS -----------

def prediction(X, betas):
    y_hat = np.dot(X, betas)
    return y_hat

def update_betas(X, Betas, y, y_hat, u):
    prediction = (y - y_hat)
    Betas_new = Betas + u*2*np.dot(X.T,prediction)
    return Betas_new

def RMSE(y_train, y_hat):
    loss = np.sqrt(sum((y_train - y_hat)**2))/len(y_train)
    return loss

def normalize_rows(x: np.ndarray):
    return x/np.linalg.norm(x, ord=2, axis=1, keepdims=True)
# source: https://necromuralist.github.io/neural_networks/posts/normalizing-with-numpy/#:~:text=Normalize%20Rows,See%20the%20numpy%20documentation.


# -----------  MPI INITIALIZATION -----------

comm = MPI.COMM_WORLD
worker = comm.Get_rank()
num_workers = comm.Get_size()
name = MPI.Get_processor_name()


initial_time = MPI.Wtime()
path = "/Users/juanfer/Documents/Maestria/SemesterIII/Lab-DDA/tutorial-4/dataset"
u = 0.0001
num_iter = 500

if worker == 0:
    data = import_data(path)
    X, y = sparse_matrix(data, path)
    X = normalize_rows(X)
    # Transform into dataframe and split into train/test - transform Numpy array
    Xdata = pd.DataFrame(X)
    ydata = pd.DataFrame(y)
    X_train = Xdata.sample(frac=0.7)
    y_train = ydata.sample(frac=0.7)
    X_test = Xdata.drop(X_train.index)
    y_test = ydata.drop(y_train.index)
    X_train = np.array(X_train)
    X_test = X_test.values
    y_train = np.array(y_train)
    y_test = y_test.values
    p_workers = round(X_train.shape[0]/(num_workers))
    X_worker = [X_train[(i*p_workers):(p_workers*(i+1))] for i in range(num_workers)]
    y_worker = [y_train[(i*p_workers):(p_workers*(i+1))] for i in range(num_workers)]
    Betas = np.random.randint(0,2,size=((X.shape[1])))

else:
    X_worker = None
    y_worker = None
    Betas = None

# Scatering X and y
X_worker = comm.scatter(X_worker, root=0)
y_worker = comm.scatter(y_worker, root=0)
betas = comm.bcast(Betas, root=0)
#print("i am worker",worker,"i received chunk:",X.shape)
rmse_train = []
rmse_test = []
timing = []
for i in range(num_iter):
    initial_time_epoch = MPI.Wtime()
    #print("i am worker",worker,"i received chunk:",type(X_worker))
    y_hat = prediction(X_worker, betas)
    betas_new = update_betas(X_worker, betas, y_worker, y_hat, u)

    comm.Barrier()
    weights = comm.reduce(betas_new, op=MPI.SUM, root=0)

    if worker == 0:
        betas_new = weights / num_workers
        y_hat_train = prediction(X_train, betas_new)
        y_hat_test = prediction(X_test, betas_new)
        rmse_tr = RMSE(y_train, y_hat_train)
        rmse_te = RMSE(y_test, y_hat_test)
        rmse_train.append(rmse_tr)
        rmse_test.append(rmse_te)
        Betas = betas_new
    betas = comm.bcast(Betas, root=0)
    
    final_time_epoch = MPI.Wtime() - initial_time
    timing.append(final_time_epoch)

print('Global Time:',MPI.Wtime() - initial_time)
plt.figure(figsize=((13,6)))
plt.plot(range(num_iter), rmse_train, marker='v', label="Train", color='blue', linewidth=3)
plt.plot(range(num_iter), rmse_test, marker='o', label="Test", color='red', linewidth=3)
plt.legend()
plt.title("VIRUS DATASET - WORKERS")
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.show()

# Save each array of testing and timing for future plotting --> See report.
pd.DataFrame(rmse_test).to_csv("/Users/juanfer/Documents/Maestria/SemesterIII/Lab-DDA/tutorial-4/virus-arrays/workers-22-test.csv")
pd.DataFrame(timing).to_csv("/Users/juanfer/Documents/Maestria/SemesterIII/Lab-DDA/tutorial-4/virus-arrays/workers-22-timing.csv")
