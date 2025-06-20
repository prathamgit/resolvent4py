from mpi4py import MPI
import psutil
from memory_profiler import profile
import sys
import os
import numpy as np
import time
import tracemalloc
import gc

sys.path.append("../../../")
import resolvent4py as res4py


def get_memory_usage(comm):
    process = psutil.Process()
    value_loc = process.memory_info().rss / (1024**2)  # Convert bytes to MB
    values = comm.allgather(value_loc)
    value = sum(values)
    return value


@profile
def run_eig(comm, op, niter):
    dmem = np.zeros(niter)
    for k in range(niter):
        mem_before = get_memory_usage(comm)
        Q, H = res4py.arnoldi_iteration(op, op.apply, 100)
        Q.destroy()
        del H
        gc.collect()
        mem_after = get_memory_usage(comm)
        dmem[k] = np.abs(mem_before - mem_after)

    avg = np.mean(dmem)
    std = np.std(dmem)
    res4py.petscprint(comm, "---------------------------------------")
    res4py.petscprint(comm, "Report for method: apply")
    res4py.petscprint(
        comm, "(Avg, Std) = (%1.4e [Mb], %1.4e [Mb])" % (avg, std)
    )
    res4py.petscprint(comm, "---------------------------------------")


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
sys.stdout = sys.__stdout__ if rank == 0 else open(os.devnull, "w")

N = 3000
Nl = res4py.compute_local_size(N)
sizes = ((Nl, N), (Nl, N))
A = res4py.generate_random_petsc_sparse_matrix(sizes, True)
ksp = res4py.create_mumps_solver(A)
op = res4py.MatrixLinearOperator(comm, A, ksp)

niter = 60

run_eig(comm, op, niter)
