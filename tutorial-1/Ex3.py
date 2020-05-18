# Tutorial 1
# Juan Fernando Espinosa
# 303158


from mpi4py import MPI
import numpy as np 

comm = MPI.COMM_WORLD
worker = comm.Get_rank()
num_workers = comm.Get_size()

N = 10**3
p_workers = round(N/num_workers)

def create_data():
    np.random.seed(3)
    A = np.random.rand(N,N)
    B = np.random.rand(N,N)
    return A, B

def matrixMult(A, B):
    initial_time = MPI.Wtime()
    C = np.zeros((A.shape[0], B.shape[0]))
    for i in range(len(A)):
        for j in range(len(B)):
            mult = 0
            for k in range(len(B[0])):
                mult += A[i][k] * B[k][j]
            C[i][j] = mult
    end_time = MPI.Wtime() - initial_time
    return C, end_time

if worker == 0:
    A, B =  create_data()
    A1 = A[:p_workers]
    for i in range(1, num_workers):
        A2 = A[(i*p_workers): (p_workers*(i+1))]
        data = (A2, B)
        comm.send(data, dest= i)
    Matrix_0, final_time = matrixMult(A1, B)

    for i in range(1, num_workers):
        output, time_worker = comm.recv()
        Matrix_0 = np.vstack((Matrix_0, output))
        final_time += time_worker
    #print('Matrix C:', Matrix_0) # Print final matrix.
    print('Time:',final_time)

else:
    A, B = comm.recv()
    matrix_worker, time_worker = matrixMult(A, B)
    info = (matrix_worker, time_worker)
    comm.send(info, dest=0)





