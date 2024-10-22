import scipy as sp
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append('../../')

import resolvent4py as res4py
from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

N = 10

path = 'data/'

fnames_jac, fnames = None, None
if rank == 0:

    os.makedirs(path) if os.path.isdir(path) == False else None
    A = sp.sparse.csr_matrix(np.random.randn(N,N) + 1j*np.random.randn(N,N))
    rows, cols = A.nonzero()
    data = A.data
    arrays = [rows,cols,data]
    fnames_jac = [path + 'rows.dat',path + 'cols.dat',path + 'vals.dat']
    for (i,array) in enumerate(arrays):
        vec = PETSc.Vec().createWithArray(array,len(array),None,MPI.COMM_SELF)
        res4py.write_to_file(MPI.COMM_SELF, fnames_jac[i], vec)
        vec.destroy()
    A = A.todense()
    u, s, v = sp.linalg.svd(sp.linalg.inv(A))
    v = v.conj().T

comm.Barrier()

fnames_jac = comm.bcast(fnames_jac, root=0)
Nl = res4py.compute_local_size(N)
A = res4py.read_coo_matrix(comm, fnames_jac, ((Nl, N),(Nl,N)))
ksp = res4py.create_mumps_solver(comm, A)
linop = res4py.MatrixLinearOperator(comm, A, ksp)

U, S, V = res4py.randomized_svd(linop, linop.solve_mat, 10, 2, 1)

Svec = S.getDiagonal()
Sseq = res4py.distributed_to_sequential_vector(comm, Svec).getArray()

if rank == 0:
    plt.figure()
    plt.plot(s.real, 'ko')
    plt.plot(Sseq.real, 'rx')
    plt.gca().set_yscale('log')
    plt.savefig("svals.png")

    print(np.abs(u[:,0]))
    print(np.abs(v[:,0]))

for i in range (S.getSizes()[-1][-1]):
    u = U.getColumnVector(i)
    v = V.getColumnVector(i)
    Au = linop.apply(u)
    error = np.abs(Au.dot(v) - 1./Sseq[i])
    res4py.petscprint(comm, "Error = %1.15e"%error)
    u.abs()
    u.view()
    v.abs()
    v.view()
    u.destroy()
    v.destroy()