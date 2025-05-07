import os
import numpy as np
import pytest
import resolvent4py as res4py
from mpi4py import MPI
from petsc4py import PETSc

@pytest.fixture(scope="module")
def comm():
    return MPI.COMM_WORLD

@pytest.fixture(scope="module")
def test_data(tmp_path_factory, comm):
    rank = comm.Get_rank()
    N, r, q, s = 100, 10, 7, 17
    path = tmp_path_factory.mktemp("low_rank_data")
    fnames_factors = None
    if rank == 0:
        U = np.random.randn(N, r) + 1j * np.random.randn(N, r)
        Sigma = np.random.randn(r, q) + 1j * np.random.randn(r, q)
        V = np.random.randn(N, q) + 1j * np.random.randn(N, q)
        A = U @ Sigma @ V.conj().T

        np.save(str(path / "Sigma.npy"), Sigma)
        Umat = PETSc.Mat().createDense((N, r), None, U, MPI.COMM_SELF)
        Vmat = PETSc.Mat().createDense((N, q), None, V, MPI.COMM_SELF)
        objs = [Umat, Vmat]
        fnames_ = ["U", "V"]
        fnames_factors = [str(path / f"{root}.dat") for root in fnames_]
        for k, obj in enumerate(objs):
            res4py.write_to_file(MPI.COMM_SELF, fnames_factors[k], obj)

        x = np.random.randn(A.shape[0]) + 1j * np.random.randn(A.shape[0])
        xvec = PETSc.Vec().createWithArray(x, comm=MPI.COMM_SELF)
        Ax = PETSc.Vec().createWithArray(A @ x, comm=MPI.COMM_SELF)
        ATx = PETSc.Vec().createWithArray(A.conj().T @ x, comm=MPI.COMM_SELF)
        X = np.random.randn(N, s) + 1j * np.random.randn(N, s)
        Xmat = PETSc.Mat().createDense((N, s), None, X, MPI.COMM_SELF)
        AX = PETSc.Mat().createDense((N, s), None, A @ X, MPI.COMM_SELF)
        ATX = PETSc.Mat().createDense((N, s), None, A.conj().T @ X, MPI.COMM_SELF)

        objs = [xvec, Ax, ATx, Xmat, AX, ATX]
        fnames_ = ["xvec", "Axvec", "ATxvec", "X", "AX", "ATX"]
        fnames = [str(path / f"{root}.dat") for root in fnames_]
        for k, obj in enumerate(objs):
            res4py.write_to_file(MPI.COMM_SELF, fnames[k], obj)

    comm.Barrier()
    return locals()

def test_low_rank_factors(comm, test_data):
    rank = comm.Get_rank()
    if rank == 0:
        # Example assertion: check shapes
        Sigma = np.load(str(test_data['path'] / "Sigma.npy"))
        assert Sigma.shape[0] == 10
        assert Sigma.shape[1] == 7
        # Add more assertions as needed
    comm.Barrier()

def test_low_rank_apply(comm, test_data):
    rank = comm.Get_rank()
    Nl = res4py.compute_local_size(test_data['N'])
    rl = res4py.compute_local_size(test_data['r'])
    ql = res4py.compute_local_size(test_data['q'])
    sl = res4py.compute_local_size(test_data['s'])
    Sig = np.load(str(test_data['path'] / "Sigma.npy"))
    U = res4py.read_bv(comm, test_data['fnames_factors'][0], ((Nl, test_data['N']), test_data['r']))
    V = res4py.read_bv(comm, test_data['fnames_factors'][1], ((Nl, test_data['N']), test_data['q']))
    linop = res4py.LowRankLinearOperator(comm, U, Sig, V)
    x = res4py.read_vector(comm, test_data['fnames'][0])
    actions = [linop.apply, linop.apply_hermitian_transpose]
    strs = ["apply", "apply_hermitian_transpose"]
    for i in range(1, 3):
        y = res4py.read_vector(comm, test_data['fnames'][i])
        y.axpy(-1.0, actions[i - 1](x))
        assert y.norm() < 1e-10
        y.destroy()
    X = res4py.read_bv(comm, test_data['fnames'][3], ((Nl, test_data['N']), test_data['s']))
    actions = [linop.apply_mat, linop.apply_hermitian_transpose_mat]
    strs = ["apply_mat", "apply_hermitian_transpose_mat"]
    for i in range(4, 6):
        Y = res4py.read_bv(comm, test_data['fnames'][i], ((Nl, test_data['N']), test_data['s']))
        res4py.bv_add(-1.0, Y, actions[i - 4](X))
        assert Y.norm() < 1e-10
        Y.destroy()
    comm.Barrier()
