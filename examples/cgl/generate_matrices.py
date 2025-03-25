import numpy as np
import scipy as sp
import cgl

import os
import sys
sys.path.append('../../')
import resolvent4py as res4py

from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

comm = MPI.COMM_WORLD

if comm.Get_size() > 1:
    raise ValueError ("This script should be run in series.")

# Define the physical parameters of the CGL equation
L = 30*2
nx = 2000
x = np.linspace(-L/2, L/2, num=nx, endpoint=True) 
nu = 1.0*(2 + 0.4*1j)
gamma = 1 - 1j
mu0 = 0.38
mu2 = -0.01
sigma = 0.4
system = cgl.CGL(x, nu, gamma, mu0, mu2, sigma)

# Save the linear operator A to file (we save the COO format vectors 
# (row indices, col indices, value))
save_path = 'data/'
os.makedirs(save_path) if not os.path.exists(save_path) else None
system.A = system.A.tocoo()
arrays = [system.A.row, system.A.col, system.A.data]
fnames = ['rows.dat', 'cols.dat', 'vals.dat']
for (i, array) in enumerate(arrays):
    fname = save_path + fnames[i]
    vec = PETSc.Vec().createWithArray(array, len(array), None, MPI.COMM_SELF)
    res4py.write_to_file(comm, fname, vec)

# Generate input and output matrices B and C as in Chen & Rowley, 2011
B = system.assemble_input_operators([-0.75, 0.75])
C = system.assemble_input_operators([-0.5, 0.5])

sizesB = ((system.nx, system.nx), (B.shape[-1], B.shape[-1]))
sizesC = ((system.nx, system.nx), (C.shape[-1], C.shape[-1]))
Bmat = PETSc.Mat().createDense(sizesB, None, B, comm)
Cmat = PETSc.Mat().createDense(sizesC, None, C, comm)

Bbv = SLEPc.BV().createFromMat(Bmat)
Bbv.setType('mat')
Cbv = SLEPc.BV().createFromMat(Cmat)
Cbv.setType('mat')

res4py.write_bv_to_file(comm, save_path + 'B.dat', Bbv)
res4py.write_bv_to_file(comm, save_path + 'C.dat', Cbv)