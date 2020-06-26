
import numpy as np 
import pandas as pd
import os
from collections import defaultdict
import time
import matplotlib.pyplot as plt

"""user_cols = ['RMSE']
df0 = pd.read_csv("/Users/juanfer/Documents/Maestria/SemesterIII/Lab-DDA/tutorial-4/kdd-arrays/no-workers-test.csv", names=user_cols)
df1 = pd.read_csv("/Users/juanfer/Documents/Maestria/SemesterIII/Lab-DDA/tutorial-4/kdd-arrays/workers-1-test.csv", names=user_cols)
df2 = pd.read_csv("/Users/juanfer/Documents/Maestria/SemesterIII/Lab-DDA/tutorial-4/kdd-arrays/workers-2-test.csv", names=user_cols)
df3 = pd.read_csv("/Users/juanfer/Documents/Maestria/SemesterIII/Lab-DDA/tutorial-4/kdd-arrays/workers-3-test.csv", names=user_cols)
df4 = pd.read_csv("/Users/juanfer/Documents/Maestria/SemesterIII/Lab-DDA/tutorial-4/kdd-arrays/workers-4-test.csv", names=user_cols)
df6 = pd.read_csv("/Users/juanfer/Documents/Maestria/SemesterIII/Lab-DDA/tutorial-4/kdd-arrays/workers-6-test.csv", names=user_cols)
df0 = df0.iloc[1:]
df1 = df1.iloc[1:]
df2 = df2.iloc[1:]
df3 = df3.iloc[1:]
df4 = df4.iloc[1:]
df6 = df6.iloc[1:]
plt.figure(figsize=((13,6)))
plt.plot(range(len(df0)), df0, label="Sequential", color='purple', linewidth=1)
plt.plot(range(len(df3)), df3, label="workers 3", color='blue', linewidth=1)
plt.plot(range(len(df1)), df1, label="workers 1", color='red', linewidth=1)
plt.plot(range(len(df2)), df2, label="workers 2", color='green', linewidth=1)
plt.plot(range(len(df4)), df4, label="workers 4", color='black', linewidth=1)
plt.plot(range(len(df6)), df6, label="workers 6", color='gray', linewidth=1)

#plt.plot(range(num_iter), rmse_test, marker='o', label="Test", color='red', linewidth=3)
plt.legend()
plt.title("VIRUS DATASET - WORKERS")
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.show()"""



user_cols = ['time']

df4 = pd.read_csv("/Users/juanfer/Documents/Maestria/SemesterIII/Lab-DDA/tutorial-4/virus-arrays/workers-4-timing.csv", names=user_cols)
df1 = pd.read_csv("/Users/juanfer/Documents/Maestria/SemesterIII/Lab-DDA/tutorial-4/virus-arrays/workers-1-timing.csv", names=user_cols)
df2 = pd.read_csv("/Users/juanfer/Documents/Maestria/SemesterIII/Lab-DDA/tutorial-4/virus-arrays/workers-2-timing.csv", names=user_cols)
df3 = pd.read_csv("/Users/juanfer/Documents/Maestria/SemesterIII/Lab-DDA/tutorial-4/virus-arrays/workers-3-timing.csv", names=user_cols)
df1 = df1.iloc[1:].sum()
df2 = df2.iloc[1:].sum()
df3 = df3.iloc[1:].sum()
df4 = df4.iloc[1:].sum()
frames = [df1, df2, df3, df4]
df = pd.concat(frames)
print(df)


plt.figure(figsize=((13,6)))
plt.plot(range(4), df, label="time", color='blue', linewidth=1)
#plt.plot(range(len(df1)), df1, label="worker 1", color='red', linewidth=1)
#plt.plot(range(len(df2)), df2, label="worker 2", color='green', linewidth=1)
#plt.plot(range(len(df4)), df4, label="worker 4", color='black', linewidth=1)


#plt.plot(range(num_iter), rmse_test, marker='o', label="Test", color='red', linewidth=3)
plt.legend()
plt.title("VIRUS DATASET - TIME VS WORKERS")
plt.xlabel("EPOCHS")
plt.ylabel("TIME")
plt.show()