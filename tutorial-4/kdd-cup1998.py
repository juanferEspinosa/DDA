from mpi4py import MPI
import numpy as np 
import pandas as pd
import os
from collections import defaultdict
import time
import matplotlib.pyplot as plt

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
# source: [1]



# -----------  MPI INITIALIZATION -----------

comm = MPI.COMM_WORLD
worker = comm.Get_rank()
num_workers = comm.Get_size()
name = MPI.Get_processor_name()

initial_time = MPI.Wtime()
u = 0.0000000001
num_iter = 100

if worker == 0:
    df = pd.read_csv("cup98LRN.txt", sep=',')
    # Select the most impactful columns to reduce the size of the dataframe (Correlation)

    features = ['TARGET_D', 'AGE', 'GENDER', 'STATE', 'MDMAUD', 'WWIIVETS','HIT', 'DATASRCE', 'VIETVETS', 'LOCALGOV', 'PVASTATE', 'STATEGOV', 'NUMPROM','RAMNTALL', 'MALEMILI','CARDPROM', 'CARDGIFT','AVGGIFT']

    #Y = df['TARGET_D']

    df = df[features]
    df= pd.get_dummies(df)
    X = df.loc[:, df.columns != 'TARGET_D']
    Y = df['TARGET_D']

    X_train = X.sample(frac=0.7)
    y_train = Y.sample(frac=0.7)
    X_test = X.drop(X_train.index)
    y_test = Y.drop(y_train.index)
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

# Scatering X and y | broadcasting betas
X_worker = comm.scatter(X_worker, root=0)
y_worker = comm.scatter(y_worker, root=0)
betas = comm.bcast(Betas, root=0)

rmse_train = []
rmse_test = []
timing = []

for i in range(num_iter):
    initial_time_epoch = MPI.Wtime()
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
plt.title("KDD DATASET - WORKERS")
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.show()

# Save each array of testing and timing for future plotting --> See report.
pd.DataFrame(rmse_test).to_csv("/Users/juanfer/Documents/Maestria/SemesterIII/Lab-DDA/tutorial-4/kdd-arrays/workers-6-test.csv")
pd.DataFrame(timing).to_csv("/Users/juanfer/Documents/Maestria/SemesterIII/Lab-DDA/tutorial-4/kdd-arrays/workers-6-timing.csv")
