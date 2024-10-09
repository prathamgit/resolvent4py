import sys
sys.path.append('../')
import numpy as np

import LinToolbox4py as lin

from mpi4py import MPI
from petsc4py import PETSc

N, Nb, Nc, r = 10, 5, 3, 2
Nl = lin.compute_local_size(N)
Nbl = lin.compute_local_size(Nb)
Ncl = lin.compute_local_size(Nc)
rl = lin.compute_local_size(r)

path = 'data/'
fnames = [path + 'rows.dat',path + 'cols.dat',path + 'vals.dat']

A = lin.read_coo_matrix(MPI.COMM_WORLD,fnames,((Nl,N),(Nl,N)))
B = lin.read_dense_matrix(MPI.COMM_WORLD,path + 'B.dat',((Nl,N),(Nbl,Nb)))
C = lin.read_dense_matrix(MPI.COMM_WORLD,path + 'C.dat',((Nl,N),(Ncl,Nc)))
K = lin.read_dense_matrix(MPI.COMM_WORLD,path + 'K.dat',((Nbl,Nb),(Ncl,Nc)))
Phi = lin.read_dense_matrix(MPI.COMM_WORLD,path + 'Phi.dat',((Nl,N),(rl,r)))
Psi = lin.read_dense_matrix(MPI.COMM_WORLD,path + 'Psi.dat',((Nl,N),(rl,r)))


lin_op = lin.LinearOperator(A,(B,K,C),(Phi,Psi,0))




x = lin.read_vector(MPI.COMM_WORLD,path + 'x.dat')
y = lin_op.apply(x)
yT = lin_op.apply(x,mode='adjoint')

ygt = lin.read_vector(MPI.COMM_WORLD,path + 'y.dat')
ygt.axpy(-1.0,y)

yTgt = lin.read_vector(MPI.COMM_WORLD,path + 'yT.dat')
yTgt.axpy(-1.0,yT)

lin.petscprint(MPI.COMM_WORLD,"Error = %1.15e"%(ygt.norm()))
lin.petscprint(MPI.COMM_WORLD,"Error = %1.15e"%(yTgt.norm()))
