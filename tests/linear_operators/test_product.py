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
def product_data(tmp_path_factory, comm):
    rank = comm.Get_rank()
    mats_names = ['A1', 'A2', 'A3', 'A4']
    row_sizes = [10, 30, 30, 20]
    col_sizes = row_sizes[1:]
    col_sizes.append(7)
    s = 9
    path = tmp_path_factory.mktemp("product_data")
    fnames = None
    if rank == 0:
        matrices = []
        for i in range(len(row_sizes)):
            Nr = row_sizes[i]
            Nc = col_sizes[i]
            A = sp.sparse.random(Nr, Nc, 0.1, "csr", np.complex128)
            if Nr == Nc:
                A += sp.sparse.identity(Nr, np.complex128, "csr")
            A = A.tocoo()
            arrays = [A.row, A.col, A.data]
            name = str(path / (mats_names[i] + '_'))
            fnames_jac = [name + "rows.dat", name + "cols.dat", name + "vals.dat"]
            for j, array in enumerate(arrays):
                vec = PETSc.Vec().createWithArray(array, len(array), None, MPI.COMM_SELF)
                res4py.write_to_file(MPI.COMM_SELF, fnames_jac[j], vec)
                vec.destroy()
            A = A.todense()
            matrices.append(A.copy())
        A = matrices[0]
        for mat in matrices[1:]:
            A = A @ mat
        x = np.random.randn(A.shape[-1]) + 1j * np.random.randn(A.shape[-1])
        xvec = PETSc.Vec().createWithArray(x, comm=MPI.COMM_SELF)
        Ax = PETSc.Vec().createWithArray(A @ x, comm=MPI.COMM_SELF)
        X = np.random.randn(A.shape[-1], s) + 1j * np.random.randn(A.shape[-1], s)
        Xmat = PETSc.Mat().createDense((A.shape[-1], s), None, X, MPI.COMM_SELF)
        AX = PETSc.Mat().createDense((A.shape[0], s), None, A @ X, MPI.COMM_SELF)
        objs = [xvec, Ax, Xmat, AX]
        fnames_ = ["xvec", "Axvec", "X", "AX"]
        fnames = [str(path / f"{root}.dat") for root in fnames_]
        for k, obj in enumerate(objs):
            res4py.write_to_file(MPI.COMM_SELF, fnames[k], obj)
    comm.Barrier()
    return {'path': path, 'fnames': comm.bcast(fnames, root=0)}

def test_product_files_exist(comm, product_data):
    rank = comm.Get_rank()
    if rank == 0:
        for fname in product_data['fnames']:
            assert os.path.exists(fname)
    comm.Barrier()

def test_product_vector(comm, product_data):
    fnames = product_data['fnames']
    lops = []
    actions = []
    for i in range(4):
        name = str(product_data['path'] / (f'A{i+1}_'))
        fnames_jac = [name + "rows.dat", name + "cols.dat", name + "vals.dat"]
        Nr = 10 if i == 0 else 30
        Nc = 30 if i < 2 else 20 if i == 2 else 7
        Nrl = res4py.compute_local_size(Nr)
        Ncl = res4py.compute_local_size(Nc)
        A = res4py.read_coo_matrix(comm, fnames_jac, ((Nrl, Nr), (Ncl, Nc)))
        Alop = res4py.MatrixLinearOperator(comm, A)
        lops.append(Alop)
        actions.append(Alop.apply)
    linop = res4py.ProductLinearOperator(comm, lops, actions)
    x = res4py.read_vector(comm, fnames[0])
    actions = [linop.apply]
    strs = ["apply"]
    for i in range(1, 2):
        y = res4py.read_vector(comm, fnames[i])
        y.axpy(-1.0, actions[i - 1](x))
        string = f"Error for {strs[i - 1]:30} = {y.norm():.15e}"
        res4py.petscprint(comm, string)
        assert y.norm() < 1e-10
        y.destroy()

def test_product_matrix(comm, product_data):
    fnames = product_data['fnames']
    lops = []
    actions = []
    for i in range(4):
        name = str(product_data['path'] / (f'A{i+1}_'))
        fnames_jac = [name + "rows.dat", name + "cols.dat", name + "vals.dat"]
        Nr = 10 if i == 0 else 30
        Nc = 30 if i < 2 else 20 if i == 2 else 7
        Nrl = res4py.compute_local_size(Nr)
        Ncl = res4py.compute_local_size(Nc)
        A = res4py.read_coo_matrix(comm, fnames_jac, ((Nrl, Nr), (Ncl, Nc)))
        Alop = res4py.MatrixLinearOperator(comm, A)
        lops.append(Alop)
        actions.append(Alop.apply)
    linop = res4py.ProductLinearOperator(comm, lops, actions)
    X = res4py.read_bv(comm, fnames[2], ((res4py.compute_local_size(20), 20), 9))
    actions = [linop.apply_mat]
    strs = ["apply_mat"]
    for i in range(1, 2):
        Y = res4py.read_bv(comm, fnames[i+2], ((res4py.compute_local_size(10), 10), 9))
        res4py.bv_add(-1.0, Y, actions[i - 1](X))
        string = f"Error for {strs[i - 1]:30} = {Y.norm():.15e}"
        res4py.petscprint(comm, string)
        assert Y.norm() < 1e-10
        Y.destroy()