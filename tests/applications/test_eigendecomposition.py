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

N = 100
s = 10

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
    evals, evecs = sp.linalg.eig(A)

comm.Barrier()

fnames_jac = comm.bcast(fnames_jac, root=0)
Nl = res4py.compute_local_size(N)
A = res4py.read_coo_matrix(comm, fnames_jac, ((Nl, N),(Nl,N)))
ksp = res4py.create_mumps_solver(comm, A)
linop = res4py.MatrixLinearOperator(comm, A, ksp)

V, D, W = res4py.right_and_left_eigendecomposition(linop, linop.solve, 100,\
                                                   10, lambda x: 1./x)

Dvec = D.getDiagonal()
Dseq = res4py.distributed_to_sequential_vector(comm, Dvec).getArray()

if rank == 0:
    plt.figure()
    plt.plot(evals.real, evals.imag, 'ko')
    plt.plot(Dseq.real, Dseq.imag, 'rx')
    plt.savefig("evals.png")

for i in range (D.getSizes()[-1][-1]):
    w = W.getColumnVector(i)
    v = V.getColumnVector(i)
    Av = linop.apply(v)
    error = np.abs(Av.dot(w) - Dseq[i])
    res4py.petscprint(comm, "Error = %1.15e"%error)
    w.destroy()
    v.destroy()