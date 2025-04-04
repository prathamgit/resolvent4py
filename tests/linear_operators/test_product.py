import scipy as sp
import numpy as np
import sys
import os

sys.path.append("../../")

import resolvent4py as res4py
from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

N = 100
s = 10

path = "data/"

fnames_jac, fnames = None, None
if rank == 0:
    os.makedirs(path) if os.path.isdir(path) == False else None
    A1 = sp.sparse.random(N, N, 0.1, "csr", np.complex128)
    A1 += sp.sparse.identity(N, np.complex128, "csr")
    A1 = A1.tocoo()
    arrays = [A1.row, A1.col, A1.data]
    fnames_jac = [path + "rows.dat", path + "cols.dat", path + "vals.dat"]
    for i, array in enumerate(arrays):
        vec = PETSc.Vec().createWithArray(
            array, len(array), None, MPI.COMM_SELF
        )
        res4py.write_to_file(MPI.COMM_SELF, fnames_jac[i], vec)
        vec.destroy()
    A1 = A1.todense()

    A2 = sp.sparse.random(N, N // 2, 0.1, "csr", np.complex128)
    A2 = A2.tocoo()
    arrays = [A2.row, A2.col, A2.data]
    fnames_jac = [path + "rows.dat", path + "cols.dat", path + "vals.dat"]
    for i, array in enumerate(arrays):
        vec = PETSc.Vec().createWithArray(
            array, len(array), None, MPI.COMM_SELF
        )
        res4py.write_to_file(MPI.COMM_SELF, fnames_jac[i], vec)
        vec.destroy()
    A2 = A2.todense()

    A = sp.sparse.csr_matrix(
        np.random.randn(N // 2, N // 4) + 1j * np.random.randn(N // 2, N // 4)
    )
    rows, cols = A.nonzero()
    data = A.data
    arrays = [rows, cols, data]
    fnames_jac = [path + "rows3.dat", path + "cols3.dat", path + "vals3.dat"]
    for i, array in enumerate(arrays):
        vec = PETSc.Vec().createWithArray(
            array, len(array), None, MPI.COMM_SELF
        )
        res4py.write_to_file(MPI.COMM_SELF, fnames_jac[i], vec)
        vec.destroy()
    A3 = A.todense()

    # A = A3@A2.conj().T@sp.linalg.inv(A1)
    A = A1 @ A2 @ A3

    x = np.random.randn(A.shape[-1]) + 1j * np.random.randn(A.shape[-1])
    xvec = PETSc.Vec().createWithArray(x, comm=MPI.COMM_SELF)
    Ax = PETSc.Vec().createWithArray(A @ x, comm=MPI.COMM_SELF)

    X = np.random.randn(A.shape[-1], s) + 1j * np.random.randn(A.shape[-1], s)
    Xmat = PETSc.Mat().createDense((A.shape[-1], s), None, X, MPI.COMM_SELF)
    AX = PETSc.Mat().createDense((A.shape[0], s), None, A @ X, MPI.COMM_SELF)

    objs = [xvec, Ax, Xmat, AX]
    fnames_ = ["x", "Ax", "X", "AX"]
    fnames = [path + root + ".dat" for root in fnames_]
    for k, obj in enumerate(objs):
        res4py.write_to_file(MPI.COMM_SELF, fnames[k], obj)

comm.Barrier()

fnames = comm.bcast(fnames, root=0)
fnames_jac1 = [path + "rows1.dat", path + "cols1.dat", path + "vals1.dat"]
fnames_jac2 = [path + "rows2.dat", path + "cols2.dat", path + "vals2.dat"]
fnames_jac3 = [path + "rows3.dat", path + "cols3.dat", path + "vals3.dat"]
fnames_vecs = fnames[:2]
fnames_mats = fnames[2:]

Nl1 = res4py.compute_local_size(N)
Nl2 = res4py.compute_local_size(N // 2)
Nl3 = res4py.compute_local_size(N // 4)
sl = res4py.compute_local_size(s)
A1 = res4py.read_coo_matrix(comm, fnames_jac1, ((Nl1, N), (Nl1, N)))
linop1 = res4py.MatrixLinearOperator(comm, A1)
A2 = res4py.read_coo_matrix(comm, fnames_jac2, ((Nl1, N), (Nl2, N // 2)))
linop2 = res4py.MatrixLinearOperator(comm, A2)
A3 = res4py.read_coo_matrix(comm, fnames_jac3, ((Nl2, N // 2), (Nl3, N // 4)))
linop3 = res4py.MatrixLinearOperator(comm, A3)
actions = [linop3.apply, linop2.apply, linop1.apply]
linop = res4py.ProductLinearOperator(comm, [linop3, linop2, linop1], actions)

x = res4py.read_vector(comm, fnames_vecs[0])
actions = [linop.apply, linop.apply]
strs = ["apply"]
for i in range(1, len(fnames_vecs)):
    y = res4py.read_vector(comm, fnames_vecs[i])
    y.axpy(-1.0, actions[i - 1](x))
    string = f"Error for {strs[i - 1]:30} = {y.norm():.15e}"
    res4py.petscprint(comm, string)
    y.destroy()
X = res4py.read_bv(comm, fnames_mats[0], ((Nl3, N // 4), s))
actions = [linop.apply_mat, linop.apply_mat]
strs = ["apply_mat"]
for i in range(1, len(fnames_vecs)):
    Y = res4py.read_bv(comm, fnames_mats[i], ((Nl1, N), s))
    res4py.bv_add(-1.0, Y, actions[i - 1](X))
    string = f"Error for {strs[i - 1]:30} = {Y.norm():.15e}"
    res4py.petscprint(comm, string)
    Y.destroy()
