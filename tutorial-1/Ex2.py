# Tutorial 1
# Juan Fernando Espinosa
# 303158


from mpi4py import MPI
import numpy as np 

comm = MPI.COMM_WORLD
worker = comm.Get_rank()
num_workers = comm.Get_size()

N = 10**4
p_workers = round(N/num_workers)

def create_data():
    np.random.seed(3)
    A = np.random.rand(N,N)
    b = np.random.rand(N,1)
    return A, b



def Matrix_vector(A, v):
    initial_time = MPI.Wtime()
    c = []
    for i in range(len(A)):
        rowA = A[i]
        for j in range(len(v[0])):
            result = 0
            for k in range(len(v)):
                numA = rowA[k]
                numB = v[k,j]
                result += numA * numB
        c.append(result)
    end_time = MPI.Wtime() - initial_time
    return c, end_time
                          
  
if worker == 0:
    A, b = create_data()
    A1 = A[:p_workers]
    for i in range(1, num_workers):
        print('i',i)
        A2 = A[(i*p_workers): (p_workers*(i+1))]
        data = (A2, b)
        comm.send(data, dest=i)
    matrix_mult, final_time = Matrix_vector(A1, b)

    c1 = []
    c1.append(matrix_mult)
    for i in range(1, num_workers):
        output, time_worker = comm.recv()
        final_time += time_worker
        c1.append(output)
    #print('Final output:', c1) # Print final vector. 
    print('time:', final_time)

    
else:
    A, b = comm.recv()
    matrix_mult, time = Matrix_vector(A, b)
    info = (matrix_mult, time)
    comm.send(info, dest=0)


    