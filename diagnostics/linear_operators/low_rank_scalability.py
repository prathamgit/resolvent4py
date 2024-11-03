import scipy as sp
import numpy as np
import sys
import os
import time as tlib

sys.path.append('../../')

import resolvent4py as res4py
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = int(1e4)
Nl = res4py.compute_local_size(N)
ru = 10
rv = 9

U = SLEPc.BV().create(comm)
U.setSizes((Nl, N), ru)
U.setType('mat')
U.setRandom()
U.orthogonalize()


V = SLEPc.BV().create(comm)
V.setSizes((Nl, N), rv)
V.setType('mat')
V.setRandom()
V.orthogonalize()


Sigma = np.random.randn(ru, rv) if rank == 0 else None
Sigma = comm.bcast(Sigma, root=0)

linop = res4py.LowRankLinearOperator(comm, U, Sigma, V)
x = PETSc.Vec().createWithArray(np.random.randn(Nl), (Nl, N), None, comm)
y = x.duplicate()


t0 = tlib.time()
for j in range (5):
    y = linop.apply(x, y)
t1 = tlib.time()
dt = np.max(np.asarray(comm.allgather((t1 - t0)/5)))
if rank == 0:
    file = open("scaling_apply.csv", "a")
    file.write("%05d, %1.5f\n"%(size, dt))
    file.close()

comm.Barrier()

t0 = tlib.time()
for j in range (5):
    y = linop.apply_hermitian_transpose(x, y)
t1 = tlib.time()
dt = np.max(np.asarray(comm.allgather((t1 - t0)/5)))
if rank == 0:
    file = open("scaling_apply_ht.csv", "a")
    file.write("%05d, %1.5f\n"%(size, dt))
    file.close()

comm.Barrier()
# res4py.petscprint(comm, "Average time for apply = %1.15f [sec]"%dt)


# res4py.petscprint(comm, \
#                 "Average time for apply_hermitian_transpose = %1.15f [sec]"%dt)

m = 100
X = SLEPc.BV().create(comm)
X.setSizes((Nl, N), 100)
X.setType('mat')
X.setRandom()
Y = X.duplicate()

t0 = tlib.time()
for j in range (10):
    Y = linop.apply_mat(X, Y)
t1 = tlib.time()
dt = np.max(np.asarray(comm.allgather((t1 - t0)/5)))
if rank == 0:
    file = open("scaling_apply_mat.csv", "a")
    file.write("%05d, %1.5f\n"%(size, dt))
    file.close()
# res4py.petscprint(comm, "Average time for apply_mat = %1.15f [sec]"%dt)
comm.Barrier()

# t0 = tlib.time()
# for j in range (10):
#     Y = linop.apply_hermitian_transpose_mat(X, Y)
# t1 = tlib.time()
# dt = np.max(np.asarray(comm.allgather((t1 - t0)/5)))
# if rank == 0:
#     file = open("scaling_apply_ht_mat.csv", "a")
#     file.write("%05d, %1.5f\n"%(size, dt))
#     file.close()
# res4py.petscprint(comm, \
#         "Average time for apply_hermitian_transpose_mat = %1.15f [sec]"%dt)

