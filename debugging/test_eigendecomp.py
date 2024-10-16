import sys
sys.path.append('../')
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import resolvent4py as res4py

from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD

N, Nb, Nc, r = 100, 5, 3, 2
Nl = res4py.compute_local_size(N)
Nbl = res4py.compute_local_size(Nb)
Ncl = res4py.compute_local_size(Nc)
rl = res4py.compute_local_size(r)

path = 'data/'
fnames = [path + 'rows.dat',path + 'cols.dat',path + 'vals.dat']

A = res4py.read_coo_matrix(comm,fnames,((Nl,N),(Nl,N)))
B = res4py.read_dense_matrix(comm,path + 'B.dat',((Nl,N),(Nbl,Nb)))
C = res4py.read_dense_matrix(comm,path + 'C.dat',((Nl,N),(Ncl,Nc)))
K = res4py.read_dense_matrix(comm,path + 'K.dat',((Nbl,Nb),(Ncl,Nc)))
Phi = res4py.read_dense_matrix(comm,path + 'Phi.dat',((Nl,N),(rl,r)))
Psi = res4py.read_dense_matrix(comm,path + 'Psi.dat',((Nl,N),(rl,r)))

ksp = res4py.create_mumps_solver(comm,A)
lin_op_ = res4py.MatrixLinearOperator(comm,A,ksp)
lin_op = res4py.ProjectedLinearOperator(comm,lin_op_,Phi,Psi,True)

evals, Q = res4py.eigendecomposition(lin_op,lin_op.apply,100,100)
# evals = (1./evals)
evals_ = np.load(path + 'evals.npy')

if comm.Get_rank() == 0:
    plt.figure()
    plt.plot(evals.real,evals.imag,'ko')
    plt.plot(evals_.real,evals_.imag,'rx')
    plt.savefig("evals.png")


_, svals, _ = res4py.randomized_svd(lin_op_,lin_op_.solve,100,1,10)
svals_ = np.load(path + 'svals.npy')
if comm.Get_rank() == 0:
    plt.figure()
    plt.plot(svals.real,'ko')
    plt.plot(svals_.real,'rx')
    plt.savefig("svals.png")


Mat = np.zeros((4,4))
Mat[1,] = 1.0
Mat[2,] = 2.0
Mat[3,] = 3.0

Mat = PETSc.Mat().createDense((4,4),None,Mat,MPI.COMM_SELF)
# Mat.view()

nloc = res4py.compute_local_size(4)
Matd = PETSc.Mat().createDense(((nloc,4),(nloc,4)),comm=MPI.COMM_WORLD)

res4py.sequential_to_distributed_matrix(Mat,Matd)
Matd.view()