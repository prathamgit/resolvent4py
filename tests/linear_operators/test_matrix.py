import pytest
import scipy as sp
import numpy as np
import os
import resolvent4py as res4py
from mpi4py import MPI
from petsc4py import PETSc

@pytest.fixture(scope="module")
def comm():
    return MPI.COMM_WORLD

@pytest.fixture(scope="module")
def matrix_data(tmp_path_factory, comm):
    rank = comm.Get_rank()
    N, s = 100, 10
    path = tmp_path_factory.mktemp("matrix_data")
    fnames_jac = fnames = None
    if rank == 0:
        A = sp.sparse.random(N, N, 0.1, "csr", np.complex128)
        A += sp.sparse.identity(N, np.complex128, "csr")
        A = A.tocoo()
        arrays = [A.row, A.col, A.data]
        fnames_jac = [str(path / "rows.dat"), str(path / "cols.dat"), str(path / "vals.dat")]
        for i, array in enumerate(arrays):
            vec = PETSc.Vec().createWithArray(array, len(array), None, MPI.COMM_SELF)
            res4py.write_to_file(MPI.COMM_SELF, fnames_jac[i], vec)
            vec.destroy()
        A = A.todense()
        Ainv = sp.linalg.inv(A)
        x = np.random.randn(A.shape[0]) + 1j * np.random.randn(A.shape[0])
        xvec = PETSc.Vec().createWithArray(x, comm=MPI.COMM_SELF)
        Ax = PETSc.Vec().createWithArray(A @ x, comm=MPI.COMM_SELF)
        ATx = PETSc.Vec().createWithArray(A.conj().T @ x, comm=MPI.COMM_SELF)
        Ainvx = PETSc.Vec().createWithArray(Ainv @ x, comm=MPI.COMM_SELF)
        AinvTx = PETSc.Vec().createWithArray(Ainv.conj().T @ x, comm=MPI.COMM_SELF)
        X = np.random.randn(N, s) + 1j * np.random.randn(N, s)
        Xmat = PETSc.Mat().createDense((N, s), None, X, MPI.COMM_SELF)
        AX = PETSc.Mat().createDense((N, s), None, A @ X, MPI.COMM_SELF)
        ATX = PETSc.Mat().createDense((N, s), None, A.conj().T @ X, MPI.COMM_SELF)
        AinvX = PETSc.Mat().createDense((N, s), None, Ainv @ X, MPI.COMM_SELF)
        AinvTX = PETSc.Mat().createDense((N, s), None, Ainv.conj().T @ X, MPI.COMM_SELF)
        objs = [xvec, Ax, ATx, Ainvx, AinvTx, Xmat, AX, ATX, AinvX, AinvTX]
        fnames_ = ["xvec", "Axvec", "ATxvec", "Ainvxvec", "AinvTxvec", "X", "AX", "ATX", "AinvX", "AinvTX"]
        fnames = [str(path / f"{root}.dat") for root in fnames_]
        for k, obj in enumerate(objs):
            res4py.write_to_file(MPI.COMM_SELF, fnames[k], obj)
    comm.Barrier()
    return {'path': path, 'fnames_jac': fnames_jac, 'fnames': fnames}

def test_matrix_files_exist(comm, matrix_data):
    rank = comm.Get_rank()
    if rank == 0:
        for fname in matrix_data['fnames_jac']:
            assert os.path.exists(fname)
        for fname in matrix_data['fnames']:
            assert os.path.exists(fname)
    comm.Barrier()

def test_matrix_operator(comm, matrix_data):
    rank = comm.Get_rank()
    N, s = 100, 10
    fnames_jac = matrix_data['fnames_jac']
    fnames = matrix_data['fnames']
    Nl = res4py.compute_local_size(N)
    sl = res4py.compute_local_size(s)
    A = res4py.read_coo_matrix(comm, fnames_jac, ((Nl, N), (Nl, N)))
    ksp = res4py.create_mumps_solver(comm, A)
    linop = res4py.MatrixLinearOperator(comm, A, ksp)
    x = res4py.read_vector(comm, fnames[0])
    actions = [
        linop.apply,
        linop.apply_hermitian_transpose,
        linop.solve,
        linop.solve_hermitian_transpose,
    ]
    strs = [
        "apply",
        "apply_hermitian_transpose",
        "solve",
        "solve_hermitian_transpose",
    ]
    for i in range(1, len(fnames[:5])):
        y = res4py.read_vector(comm, fnames[i])
        y.axpy(-1.0, actions[i - 1](x))
        assert y.norm() < 1e-10
        y.destroy()
    X = res4py.read_bv(comm, fnames[5], ((Nl, N), s))
    actions = [
        linop.apply_mat,
        linop.apply_hermitian_transpose_mat,
        linop.solve_mat,
        linop.solve_hermitian_transpose_mat,
    ]
    strs = [
        "apply_mat",
        "apply_hermitian_transpose_mat",
        "solve_mat",
        "solve_hermitian_transpose_mat",
    ]
    for i in range(1, len(fnames[5:])):
        Y = res4py.read_bv(comm, fnames[i+5], ((Nl, N), s))
        res4py.bv_add(-1.0, Y, actions[i - 1](X))
        assert Y.norm() < 1e-10
        Y.destroy()
    comm.Barrier()
