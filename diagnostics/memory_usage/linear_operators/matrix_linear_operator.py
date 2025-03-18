from mpi4py import MPI
import psutil
from memory_profiler import profile
import sys
import os
import numpy as np
import time
import tracemalloc

sys.path.append('../../../')
import resolvent4py as res4py

def get_memory_usage(comm):
    process = psutil.Process()
    value_loc = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
    values = comm.allgather(value_loc)
    value = sum(values)
    return value

@profile
def run_vec_methods(comm, niter, op, names, methods):
    for (k, m) in enumerate(methods):
        if names[k] == 'apply' or names[k] == 'solve':
            x = res4py.generate_random_petsc_vector(comm, op._dimensions[-1])
            y = op.create_left_vector()
        else:
            x = res4py.generate_random_petsc_vector(comm, op._dimensions[0])
            y = op.create_right_vector()
        
        dmem = np.zeros(niter)
        for i in range (niter):
            mem_before = get_memory_usage(comm)
            y = m(x, y)
            mem_after = get_memory_usage(comm)
            dmem[i] = np.abs(mem_after - mem_before)
            
        x.destroy()
        y.destroy()
        
        avg = np.mean(dmem)
        std = np.std(dmem)
        res4py.petscprint(comm, "---------------------------------------")
        res4py.petscprint(comm, "Report for method: %s"%names[k])
        res4py.petscprint(comm, "(Avg, Std) = (%1.4e [Mb], %1.4e [Mb])"%(avg, std))
        res4py.petscprint(comm, "---------------------------------------")

@profile
def run_bv_methods(comm, niter, ncols, op, names, methods):
    for (k, m) in enumerate(methods):
        if names[k] == 'apply_mat' or names[k] == 'solve_mat':
            x = op.create_right_bv(ncols)
            y = op.create_left_bv(ncols)
        else:
            x = op.create_left_bv(ncols)
            y = op.create_right_bv(ncols)
        
        dmem = np.zeros(niter)
        for i in range (niter):
            mem_before = get_memory_usage(comm)
            y = m(x, y)
            mem_after = get_memory_usage(comm)
            dmem[i] = np.abs(mem_after - mem_before)
        
        x.destroy()
        y.destroy()
        
        avg = np.mean(dmem)
        std = np.std(dmem)
        res4py.petscprint(comm, "---------------------------------------")
        res4py.petscprint(comm, "Report for method: %s"%names[k])
        res4py.petscprint(comm, "(Avg, Std) = (%1.4e [Mb], %1.4e [Mb])"%(avg, std))
        res4py.petscprint(comm, "---------------------------------------")


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
sys.stdout = sys.__stdout__ if rank == 0 else open(os.devnull, 'w')

N = 1000
Nl = res4py.compute_local_size(N)
sizes = ((Nl, N), (Nl, N))
A = res4py.generate_random_petsc_sparse_matrix(comm, sizes, True)
ksp = res4py.create_mumps_solver(comm, A)
op = res4py.MatrixLinearOperator(comm, A, ksp)

niter = 30

methods = [op.apply, op.solve, op.apply_hermitian_transpose, \
           op.solve_hermitian_transpose]
names = ['apply', 'solve', 'apply_ht', 'solve_ht']
run_vec_methods(comm, niter, op, names, methods)

methods = [op.apply_mat, op.solve_mat, op.apply_hermitian_transpose_mat, \
           op.solve_hermitian_transpose_mat]
names = ['apply_mat', 'solve_mat', 'apply_ht_mat', 'solve_ht_mat']
run_bv_methods(comm, niter, 30, op, names, methods)


# from mpi4py import MPI
# import tracemalloc
# import sys
# import os
# import numpy as np
# import time

# sys.path.append('../../../')
# import resolvent4py as res4py

# def get_memory_usage(comm):
#     # Take a snapshot of memory allocations
#     snapshot = tracemalloc.take_snapshot()
#     # Retrieve the top memory blocks for each process
#     top_stats = snapshot.statistics('lineno')
#     total_memory = sum(stat.size for stat in top_stats) / (1024 ** 2)  # Convert bytes to MB
#     values = comm.allgather(total_memory)
#     value = sum(values)
#     return value

# def run_vec_methods(comm, niter, op, names, methods):
#     for (k, m) in enumerate(methods):
#         if names[k] == 'apply' or names[k] == 'solve':
#             x = res4py.generate_random_petsc_vector(comm, op._dimensions[-1])
#             y = op.create_left_vector()
#         else:
#             x = res4py.generate_random_petsc_vector(comm, op._dimensions[0])
#             y = op.create_right_vector()
        
#         mem_vec = x.getSizes()[-1] * 16 / (1024 ** 2)
#         dmem = np.zeros(niter)
#         for i in range(niter):
#             mem_before = get_memory_usage(comm)
#             y = m(x, y)
#             mem_after = get_memory_usage(comm)
#             dmem[i] = 100 * np.abs(mem_after - mem_before) / mem_vec
        
#         x.destroy()
#         y.destroy()
        
#         avg = np.mean(dmem)
#         std = np.std(dmem)
#         res4py.petscprint(comm, "---------------------------------------")
#         res4py.petscprint(comm, "Report for method: %s" % names[k])
#         res4py.petscprint(comm, "(Avg, Std) = (%1.4e [Mb], %1.4e [Mb])" % (avg, std))
#         res4py.petscprint(comm, "---------------------------------------")

# def run_bv_methods(comm, niter, ncols, op, names, methods):
#     for (k, m) in enumerate(methods):
#         if names[k] == 'apply_mat' or names[k] == 'solve_mat':
#             x = op.create_right_bv(ncols)
#             y = op.create_left_bv(ncols)
#         else:
#             x = op.create_left_bv(ncols)
#             y = op.create_right_bv(ncols)
        
#         sz = x.getSizes()
#         mem_bv = sz[0][-1] * sz[-1] * 16 / (1024 ** 2)
#         dmem = np.zeros(niter)
#         for i in range(niter):
#             mem_before = get_memory_usage(comm)
#             y = m(x, y)
#             mem_after = get_memory_usage(comm)
#             dmem[i] = 100 * np.abs(mem_after - mem_before) / mem_bv
        
#         x.destroy()
#         y.destroy()
        
#         avg = np.mean(dmem)
#         std = np.std(dmem)
#         res4py.petscprint(comm, "---------------------------------------")
#         res4py.petscprint(comm, "Report for method: %s" % names[k])
#         res4py.petscprint(comm, "(Avg, Std) = (%1.4e [Mb], %1.4e [Mb])" % (avg, std))
#         res4py.petscprint(comm, "---------------------------------------")

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# sys.stdout = sys.__stdout__ if rank == 0 else open(os.devnull, 'w')

# # Start tracing memory allocations
# tracemalloc.start()

# N = 1000
# Nl = res4py.compute_local_size(N)
# sizes = ((Nl, N), (Nl, N))
# A = res4py.generate_random_petsc_sparse_matrix(comm, sizes, True)
# ksp = res4py.create_mumps_solver(comm, A)
# op = res4py.MatrixLinearOperator(comm, A, ksp)

# niter = 30

# methods = [op.apply, op.solve, op.apply_hermitian_transpose, 
#            op.solve_hermitian_transpose]
# names = ['apply', 'solve', 'apply_ht', 'solve_ht']
# run_vec_methods(comm, niter, op, names, methods)

# methods = [op.apply_mat, op.solve_mat, op.apply_hermitian_transpose_mat, 
#            op.solve_hermitian_transpose_mat]
# names = ['apply_mat', 'solve_mat', 'apply_ht_mat', 'solve_ht_mat']
# run_bv_methods(comm, niter, 30, op, names, methods)

# # Stop tracing memory allocations
# tracemalloc.stop()
