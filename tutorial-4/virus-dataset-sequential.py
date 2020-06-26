import numpy as np 
import pandas as pd
import os
from collections import defaultdict
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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
initial_time = time.time()
path = "/Users/juanfer/Documents/Maestria/SemesterIII/Lab-DDA/tutorial-4/dataset"
data = import_data(path)
X, y = sparse_matrix(data, path)

# Normalization
X = normalize_rows(X)


# Transform into dataframe and split into train/test
Xdata = pd.DataFrame(X)
ydata = pd.DataFrame(y)
X_train = Xdata.sample(frac=0.7)
y_train = ydata.sample(frac=0.7)
X_test = Xdata.drop(X_train.index)
y_test = ydata.drop(y_train.index)
X_train = np.array(X_train)
X_test = X_test.values
y_train = np.array(y_train).ravel()
y_test = y_test.values.ravel()

u =0.00001
num_iter = 100
rmse_global = []
rmse_global2 = []
Betas = np.random.randint(0,2,size=((X.shape[1])))
for i in range(num_iter):
    y_hat = prediction(X_train, Betas)
    #print(y_hat)
    betas_new = update_betas(X_train, Betas, y_train, y_hat, u)
    #print(betas_new)
    loss = RMSE(y_train, y_hat)
    #print(loss)
    rmse_global.append(loss)
    Betas = betas_new
    y_hat_test = prediction(X_test, Betas)
    loss_test = RMSE(y_test, y_hat_test)
    rmse_global2.append(loss_test)
end_time = time.time() - initial_time
print('end time', end_time)

plt.figure(figsize=((13,6)))
plt.plot(range(num_iter), rmse_global, marker='v', label="Train", color='blue', linewidth=3)
plt.plot(range(num_iter), rmse_global2, marker='o', label="Test", color='red', linewidth=3)
plt.legend()
plt.title("VIRUS DATASET - NO WORKERS")
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.show()
#pd.DataFrame(rmse_global2).to_csv("/Users/juanfer/Documents/Maestria/SemesterIII/Lab-DDA/tutorial-4/virus-arrays/no-workers2-test.csv")
