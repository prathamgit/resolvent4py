import scipy as sp
import numpy as np
import sys
import os

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
    fnames_jac = [path + 'rows1.dat',path + 'cols1.dat',path + 'vals1.dat']
    for (i,array) in enumerate(arrays):
        vec = PETSc.Vec().createWithArray(array,len(array),None,MPI.COMM_SELF)
        res4py.write_to_file(MPI.COMM_SELF, fnames_jac[i], vec)
        vec.destroy()
    A1 = A.todense()

    os.makedirs(path) if os.path.isdir(path) == False else None
    A = sp.sparse.csr_matrix(np.random.randn(N,N) + 1j*np.random.randn(N,N))
    rows, cols = A.nonzero()
    data = A.data
    arrays = [rows,cols,data]
    fnames_jac = [path + 'rows2.dat',path + 'cols2.dat',path + 'vals2.dat']
    for (i,array) in enumerate(arrays):
        vec = PETSc.Vec().createWithArray(array,len(array),None,MPI.COMM_SELF)
        res4py.write_to_file(MPI.COMM_SELF, fnames_jac[i], vec)
        vec.destroy()
    A2 = A.todense()

    os.makedirs(path) if os.path.isdir(path) == False else None
    A = sp.sparse.csr_matrix(np.random.randn(N,N) + 1j*np.random.randn(N,N))
    rows, cols = A.nonzero()
    data = A.data
    arrays = [rows,cols,data]
    fnames_jac = [path + 'rows3.dat',path + 'cols3.dat',path + 'vals3.dat']
    for (i,array) in enumerate(arrays):
        vec = PETSc.Vec().createWithArray(array,len(array),None,MPI.COMM_SELF)
        res4py.write_to_file(MPI.COMM_SELF, fnames_jac[i], vec)
        vec.destroy()
    A3 = A.todense()


    A = A3@A2.conj().T@sp.linalg.inv(A1)
    Ainv = sp.linalg.inv(A)

    x = np.random.randn(A.shape[0]) + 1j*np.random.randn(A.shape[0]) 
    xvec = PETSc.Vec().createWithArray(x, comm=MPI.COMM_SELF)
    Ax = PETSc.Vec().createWithArray(A@x, comm=MPI.COMM_SELF)
    ATx = PETSc.Vec().createWithArray(A.conj().T@x, comm=MPI.COMM_SELF)
    # Ainvx = PETSc.Vec().createWithArray(Ainv@x, comm=MPI.COMM_SELF)
    # AinvTx = PETSc.Vec().createWithArray(Ainv.conj().T@x, comm=MPI.COMM_SELF)

    X = np.random.randn(N,s) + 1j*np.random.randn(N,s)
    Xmat = PETSc.Mat().createDense((N, s), None, X, MPI.COMM_SELF)
    AX = PETSc.Mat().createDense((N, s), None, A@X, MPI.COMM_SELF)
    ATX = PETSc.Mat().createDense((N, s), None, A.conj().T@X, MPI.COMM_SELF)
    AinvX = PETSc.Mat().createDense((N, s), None, Ainv@X, MPI.COMM_SELF)
    AinvTX = PETSc.Mat().createDense((N, s), None, Ainv.conj().T@X, \
                                     MPI.COMM_SELF)
    
    objs = [xvec, Ax, ATx, Xmat, AX, ATX]
    fnames_ = ['x', 'Ax', 'ATx', 'X', 'AX', 'ATX']
    fnames = [path + root + '.dat' for root in fnames_]
    for (k, obj) in enumerate(objs):
        res4py.write_to_file(MPI.COMM_SELF, fnames[k], obj)

comm.Barrier()

fnames = comm.bcast(fnames, root=0)
fnames_jac1 = [path + 'rows1.dat',path + 'cols1.dat',path + 'vals1.dat']
fnames_jac2 = [path + 'rows2.dat',path + 'cols2.dat',path + 'vals2.dat']
fnames_jac3 = [path + 'rows3.dat',path + 'cols3.dat',path + 'vals3.dat']
fnames_vecs = fnames[:3]
fnames_mats = fnames[3:]

Nl = res4py.compute_local_size(N)
sl = res4py.compute_local_size(s)
A1 = res4py.read_coo_matrix(comm, fnames_jac1, ((Nl, N),(Nl,N)))
ksp1 = res4py.create_mumps_solver(comm, A1)
linop1 = res4py.MatrixLinearOperator(comm, A1, ksp1)
A2 = res4py.read_coo_matrix(comm, fnames_jac2, ((Nl, N),(Nl,N)))
ksp2 = res4py.create_mumps_solver(comm, A2)
linop2 = res4py.MatrixLinearOperator(comm, A2, ksp2)
A3 = res4py.read_coo_matrix(comm, fnames_jac3, ((Nl, N),(Nl,N)))
ksp3 = res4py.create_mumps_solver(comm, A3)
linop3 = res4py.MatrixLinearOperator(comm, A3, ksp3)
actions = [linop1.solve, linop2.apply_hermitian_transpose, linop3.apply]
linop = res4py.ProductLinearOperator(comm, [linop1, linop2, linop3], actions)

x = res4py.read_vector(comm, fnames_vecs[0])
actions = [linop.apply, linop.apply_hermitian_transpose]
strs = ['apply', 'apply_hermitian_transpose']
for i in range (1,len(fnames_vecs)):
    y = res4py.read_vector(comm, fnames_vecs[i])
    y.axpy(-1.0, actions[i-1](x))
    string = f"Error for {strs[i-1]:30} = {y.norm():.15e}"
    res4py.petscprint(comm, string)
    y.destroy()
X = res4py.read_bv(comm, fnames_mats[0], ((Nl, N), s))
actions = [linop.apply_mat, linop.apply_hermitian_transpose_mat]
strs = ['apply_mat', 'apply_hermitian_transpose_mat']
for i in range (1,len(fnames_vecs)):
    Y = res4py.read_bv(comm, fnames_mats[i], ((Nl, N), s))
    res4py.bv_add(-1.0, Y, actions[i-1](X))
    string = f"Error for {strs[i-1]:30} = {Y.norm():.15e}"
    res4py.petscprint(comm, string)
    Y.destroy()