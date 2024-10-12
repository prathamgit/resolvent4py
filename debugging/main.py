import sys
sys.path.append('../')
import numpy as np
import scipy as sp

import resolvent4py as res4py

from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD

N, Nb, Nc, r = 10, 5, 3, 2
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
lin_op__ = res4py.LowRankUpdatedLinearOperator(comm,lin_op_,B,K,C,None)
lin_op = res4py.ProjectedLinearOperator(comm,lin_op__,Phi,Psi,False)


# res4py_op = res4py.LinearOperator(comm,A,(Phi,Psi,1),(B,K,C),None)

x = res4py.read_vector(comm,path + 'x.dat')
y = lin_op.apply(x)
yT = lin_op.apply_hermitian_transpose(x)
ygt = res4py.read_vector(comm,path + 'y.dat')
ygt.axpy(-1.0,y)
yTgt = res4py.read_vector(comm,path + 'yT.dat')
yTgt.axpy(-1.0,yT)


yinv = lin_op.solve(x)
yinvgt = res4py.read_vector(comm,path + 'yinv.dat')
yinvgt.axpy(-1.0,yinv)


yinvT = lin_op.solve_hermitian_transpose(x)
yinvTgt = res4py.read_vector(comm,path + 'yinvT.dat')
yinvTgt.axpy(-1.0,yinvT)


res4py.petscprint(comm,"Error = %1.15e"%(ygt.norm()))
res4py.petscprint(comm,"Error = %1.15e"%(yTgt.norm()))
res4py.petscprint(comm,"Error = %1.15e"%(yinvgt.norm()))
res4py.petscprint(comm,"Error = %1.15e"%(yinvTgt.norm()))


# res4py_op.destroy()