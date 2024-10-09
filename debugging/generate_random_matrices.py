import numpy as np
import scipy as sp
from mpi4py import MPI
from petsc4py import PETSc

import sys
sys.path.append('../')

import LinToolbox4py as lin

path = 'data/'

N, Nb, Nc, r = 10, 5, 3, 2

A = sp.sparse.csr_matrix(np.random.randn(N,N) + 1j*np.random.randn(N,N))
B = np.random.randn(N,Nb) + 1j*np.random.randn(N,Nb)
C = np.random.randn(N,Nc) + 1j*np.random.randn(N,Nc)
K = np.random.randn(Nb,Nc) + 1j*np.random.randn(Nb,Nc)
Phi = np.random.randn(N,r) + 1j*np.random.randn(N,r)
Psi = np.random.randn(N,r) + 1j*np.random.randn(N,r)

# Save A matrix in COO format
rows, cols = A.nonzero()
data = A.data

arrays = [rows,cols,data]
fnames = [path + 'rows.dat',path + 'cols.dat',path + 'vals.dat']
viewer = PETSc.Viewer().createBinary(fnames[0],"w",comm=MPI.COMM_SELF)
for (i,array) in enumerate(arrays):
    vec = PETSc.Vec().createWithArray(array,len(array),None,MPI.COMM_SELF)
    viewer.setFileName(fnames[i])
    vec.view(viewer)
    vec.destroy()
viewer.destroy()

# Save B, C and K matrices
arrays = [B,C,K,Phi,Psi]
fnames = [path + 'B.dat',path + 'C.dat',path + 'K.dat',path + 'Phi.dat',path + 'Psi.dat']
viewer = PETSc.Viewer().createBinary(fnames[0],"w",comm=MPI.COMM_SELF)
for (i,array) in enumerate(arrays):
    Mat = PETSc.Mat().createDense(array.shape,None,array,MPI.COMM_SELF)
    viewer.setFileName(fnames[i])
    Mat.view(viewer)
    Mat.destroy()
viewer.destroy()

Phi = Phi@sp.linalg.inv(Psi.conj().T@Phi)
P = Phi@Psi.conj().T

# Generate random vector
x = np.random.randn(N)
M = A + B@K@C.conj().T
Mat = P@M@P
y = Mat@x
yT = Mat.conj().T@x

Mat = P@sp.linalg.inv(M)@P
yinv = Mat@x
yinvT = Mat.conj().T@x

x = PETSc.Vec().createWithArray(x,N,None,comm=MPI.COMM_WORLD)
y = PETSc.Vec().createWithArray(y,N,None,comm=MPI.COMM_WORLD)
yT = PETSc.Vec().createWithArray(yT,N,None,comm=MPI.COMM_WORLD)
yinv = PETSc.Vec().createWithArray(yinv,N,None,comm=MPI.COMM_WORLD)
yinvT = PETSc.Vec().createWithArray(yinvT,N,None,comm=MPI.COMM_WORLD)

lin.write_vector(MPI.COMM_WORLD,path + 'x.dat',x)
lin.write_vector(MPI.COMM_WORLD,path + 'y.dat',y)
lin.write_vector(MPI.COMM_WORLD,path + 'yT.dat',yT)
lin.write_vector(MPI.COMM_WORLD,path + 'yinv.dat',yinv)
lin.write_vector(MPI.COMM_WORLD,path + 'yinvT.dat',yinvT)

