# Tutorial 1
# Juan Fernando Espinosa
# 303158


from mpi4py import MPI
import numpy as np 

comm = MPI.COMM_WORLD
worker = comm.Get_rank()
num_workers = comm.Get_size()
name = MPI.Get_processor_name()

N = 10**12
p_workers = round(N/num_workers)


def create_vectors():
    np.random.seed(3)
    v = np.random.rand(N)
    w = np.random.rand(N)
    data = (v,w)
    return data

# Function where the addition of row-wise elements occur 
# Time checking process.
def add_vectors(data):
    initial_time = MPI.Wtime()
    a, b = data
    z = []
    suma = [a[i]+b[i] for i in range(len(a))] 
    z.append(suma)
    end_time = MPI.Wtime() - initial_time
    return z, end_time


if worker == 0:
    a,b = create_vectors()
    a0 = a[:p_workers]
    b0 = b[:p_workers]
    data1 = (a0,b0)

    # Ideally, splitting the vectors in equal size through the workers.
    for i in range(1, num_workers):
        print('i',i)
        a1 = a[(i*p_workers):(p_workers*(i+1))]
        b1 = b[(i*p_workers):(p_workers*(i+1))]
        data = (a1,b1)
        comm.send(data, dest=i)
    final_output = []
    adding, time = add_vectors(data1)
    final_output.append(adding)
    for i in range(1, num_workers):
        output, time_worker = comm.recv()
        final_output.append(output)
        time += time_worker
    #print('final array', final_output) # Print of the final array
    print('Final time', time)

else:
    data = comm.recv(source=0)
    adding, time = add_vectors(data)
    info = (adding, time)
    comm.send(info, dest=0) 
    
