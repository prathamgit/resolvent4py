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
nb = 10
nc = 10
omega = 1.7
s = 5

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

    B = np.random.randn(N, nb) + 1j*np.random.randn(N, nb)
    Wb = np.random.randn(nb, nb) + 1j*np.random.randn(nb, nb)
    Wb = Wb@Wb.conj().T
    B_ = PETSc.Mat().createDense((N,nb), None, B, MPI.COMM_SELF)
    Wb_ = PETSc.Mat().createDense((nb,nb), None, Wb, MPI.COMM_SELF)
    res4py.write_to_file(MPI.COMM_SELF, path + 'B.dat',B_)
    res4py.write_to_file(MPI.COMM_SELF, path + 'Wb.dat',Wb_)

    C = np.random.randn(N, nc) + 1j*np.random.randn(N, nc)
    Wc = np.random.randn(nc, nc) + 1j*np.random.randn(nc, nc)
    Wc = Wc@Wc.conj().T
    C_ = PETSc.Mat().createDense((N,nc), None, C, MPI.COMM_SELF)
    Wc_ = PETSc.Mat().createDense((nc,nc), None, Wc, MPI.COMM_SELF)
    res4py.write_to_file(MPI.COMM_SELF, path + 'C.dat',C_)
    res4py.write_to_file(MPI.COMM_SELF, path + 'Wc.dat',Wc_)

    R = sp.linalg.inv(1j*omega*np.eye(N) - A)
    M = (C@Wc).conj().T@R@B@Wb

    x = np.random.randn(nb) + 1j*np.random.randn(nb) 
    xvec = PETSc.Vec().createWithArray(x, comm=MPI.COMM_SELF)
    Mx = PETSc.Vec().createWithArray(M@x, comm=MPI.COMM_SELF)
    MTx = PETSc.Vec().createWithArray(M.conj().T@x, comm=MPI.COMM_SELF)

    X = np.random.randn(nb,s) + 1j*np.random.randn(nb,s)
    Xmat = PETSc.Mat().createDense((nb, s), None, X, MPI.COMM_SELF)
    MX = PETSc.Mat().createDense((nc, s), None, M@X, MPI.COMM_SELF)
    MTX = PETSc.Mat().createDense((nb, s), None, M.conj().T@X, MPI.COMM_SELF)
    
    objs = [xvec, Mx, MTx, Xmat, MX, MTX]
    fnames_ = ['x', 'Mx', 'MTx', 'X', 'MX', 'MTX']
    fnames = [path + root + '.dat' for root in fnames_]
    for (k, obj) in enumerate(objs):
        res4py.write_to_file(MPI.COMM_SELF, fnames[k], obj)

comm.Barrier()

fnames = comm.bcast(fnames, root=0)
fnames_jac = comm.bcast(fnames_jac, root=0)
fnames_vecs = fnames[:3]
fnames_mats = fnames[3:]

Nl = res4py.compute_local_size(N)
A = res4py.read_coo_matrix(comm, fnames_jac, ((Nl, N),(Nl,N)))
Rinv = res4py.create_AIJ_identity(comm, A.getSizes())
Rinv.scale(1j*omega)
Rinv.axpy(-1.0, A)
ksp = res4py.create_mumps_solver(comm, Rinv)
Rinv_linop = res4py.MatrixLinearOperator(comm, Rinv, ksp)

nbl = res4py.compute_local_size(nb)
B = res4py.read_dense_matrix(comm, path + 'B.dat', ((Nl, N),(nbl, nb)))
Wb = res4py.read_dense_matrix(comm, path + 'Wb.dat', ((nbl, nb),(nbl, nb)))
Btil = B.matMult(Wb)
Btil_linop = res4py.MatrixLinearOperator(comm, Btil)

ncl = res4py.compute_local_size(nc)
C = res4py.read_dense_matrix(comm, path + 'C.dat', ((Nl, N),(ncl, nc)))
Wc = res4py.read_dense_matrix(comm, path + 'Wc.dat', ((ncl, nc),(ncl, nc)))
Ctil = C.matMult(Wc)
Ctil_linop = res4py.MatrixLinearOperator(comm, Ctil)

linop = res4py.InputOutputLinearOperator(comm, Rinv_linop, Btil_linop, \
                                         Ctil_linop)

sl = res4py.compute_local_size(s)
x = res4py.read_vector(comm, fnames_vecs[0])
actions = [linop.apply, linop.apply_hermitian_transpose]
strs = ['apply', 'apply_hermitian_transpose']
for i in range (1,len(fnames_vecs)):
    y = res4py.read_vector(comm, fnames_vecs[i])
    y.axpy(-1.0, actions[i-1](x))
    string = f"Error for {strs[i-1]:30} = {y.norm():.15e}"
    res4py.petscprint(comm, string)
    y.destroy()
X = res4py.read_dense_matrix(comm, fnames_mats[0], ((nbl, nb), (sl, s)))
actions = [linop.apply_mat, linop.apply_hermitian_transpose_mat]
strs = ['apply_mat', 'apply_hermitian_transpose_mat']
for i in range (1,len(fnames_vecs)):
    Y = res4py.read_dense_matrix(comm, fnames_mats[i], ((ncl, nc), (sl, s)))
    Y.axpy(-1.0, actions[i-1](X))
    string = f"Error for {strs[i-1]:30} = {Y.norm():.15e}"
    res4py.petscprint(comm, string)
    Y.destroy()