import sys
sys.path.append('../')
import numpy as np

import LinToolbox4py as lin

from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD

N, Nb, Nc, r = 10, 5, 3, 2
Nl = lin.compute_local_size(N)
Nbl = lin.compute_local_size(Nb)
Ncl = lin.compute_local_size(Nc)
rl = lin.compute_local_size(r)

path = 'data/'
fnames = [path + 'rows.dat',path + 'cols.dat',path + 'vals.dat']

A = lin.read_coo_matrix(comm,fnames,((Nl,N),(Nl,N)))
B = lin.read_dense_matrix(comm,path + 'B.dat',((Nl,N),(Nbl,Nb)))
C = lin.read_dense_matrix(comm,path + 'C.dat',((Nl,N),(Ncl,Nc)))
K = lin.read_dense_matrix(comm,path + 'K.dat',((Nbl,Nb),(Ncl,Nc)))
Phi = lin.read_dense_matrix(comm,path + 'Phi.dat',((Nl,N),(rl,r)))
Psi = lin.read_dense_matrix(comm,path + 'Psi.dat',((Nl,N),(rl,r)))


lin_op = lin.LinearOperator(comm,A,(B,K,C),(Phi,Psi,0))
lin_solver = lin.LinearSolver(lin_op)


x = lin.read_vector(comm,path + 'x.dat')
y = lin_op.apply(x)
yT = lin_op.apply(x,mode='adjoint')

ygt = lin.read_vector(comm,path + 'y.dat')
ygt.axpy(-1.0,y)

yTgt = lin.read_vector(comm,path + 'yT.dat')
yTgt.axpy(-1.0,yT)

yinv = lin_solver.solve(x)
yinvgt = lin.read_vector(comm,path + 'yinv.dat')
yinvgt.axpy(-1.0,yinv)


yinvT = lin_solver.solve(x,mode='adjoint')
yinvTgt = lin.read_vector(comm,path + 'yinvT.dat')
yinvTgt.axpy(-1.0,yinvT)

lin.petscprint(comm,"Error = %1.15e"%(ygt.norm()))
lin.petscprint(comm,"Error = %1.15e"%(yTgt.norm()))
lin.petscprint(comm,"Error = %1.15e"%(yinvgt.norm()))
lin.petscprint(comm,"Error = %1.15e"%(yinvTgt.norm()))
