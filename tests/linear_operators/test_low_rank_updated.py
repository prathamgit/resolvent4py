import os
import sys
import pytest
import numpy as np
import scipy as sp
import resolvent4py as res4py
from mpi4py import MPI
from petsc4py import PETSc
from .. import pytest_utils


def test_low_rank_updated_on_vectors(comm, square_random_matrix):
    r"""Test LowRankUpdatedLinearOperator on vectors"""

    Apetsc, Apython = square_random_matrix
    N = Apython.shape[0]
    rr, rc = 5, 9
    U, Upython = pytest_utils.generate_random_bv(comm, (N, rr))
    V, Vpython = pytest_utils.generate_random_bv(comm, (N, rc))
    S = np.random.randn(rr, rc) + 1j * np.random.randn(rr, rc)
    S = comm.bcast(S, root=0)
    Apython += Upython @ S @ Vpython.conj().T
    ksp = res4py.create_mumps_solver(comm, Apetsc)
    linop1 = res4py.linear_operators.MatrixLinearOperator(comm, Apetsc, ksp)
    linop = res4py.linear_operators.LowRankUpdatedLinearOperator(
        comm, linop1, U, S, V
    )

    x, xpython = pytest_utils.generate_random_vector(comm, N)
    Apython_inv = sp.linalg.inv(Apython)
    actions_python = [
        Apython.dot,
        Apython.conj().T.dot,
        Apython_inv.dot,
        Apython_inv.conj().T.dot,
    ]
    actions_petsc = [
        linop.apply,
        linop.apply_hermitian_transpose,
        linop.solve,
        linop.solve_hermitian_transpose,
    ]

    y = linop.create_left_vector()
    error_vec = [
        pytest_utils.compute_error_vector(
            comm, actions_petsc[i], x, y, actions_python[i], xpython
        )
        for i in range(len(actions_petsc))
    ]
    error = np.linalg.norm(error_vec)
    x.destroy()
    y.destroy()
    linop.destroy()
    assert error < 1e-8


def test_low_rank_updated_on_bvs(comm, square_random_matrix):
    r"""Test LowRankUpdatedLinearOperator on BVs"""

    Apetsc, Apython = square_random_matrix
    N = Apython.shape[0]
    rr, rc = 5, 9
    U, Upython = pytest_utils.generate_random_bv(comm, (N, rr))
    V, Vpython = pytest_utils.generate_random_bv(comm, (N, rc))
    S = np.random.randn(rr, rc) + 1j * np.random.randn(rr, rc)
    S = comm.bcast(S, root=0)
    Apython += Upython @ S @ Vpython.conj().T
    ksp = res4py.create_mumps_solver(comm, Apetsc)
    linop1 = res4py.linear_operators.MatrixLinearOperator(comm, Apetsc, ksp)
    linop = res4py.linear_operators.LowRankUpdatedLinearOperator(
        comm, linop1, U, S, V
    )

    s = 5
    X, Xpython = pytest_utils.generate_random_bv(comm, (N, s))
    Apython_inv = sp.linalg.inv(Apython)
    actions_python = [
        Apython.dot,
        Apython.conj().T.dot,
        Apython_inv.dot,
        Apython_inv.conj().T.dot,
    ]
    actions_petsc = [
        linop.apply_mat,
        linop.apply_hermitian_transpose_mat,
        linop.solve_mat,
        linop.solve_hermitian_transpose_mat,
    ]

    Y = linop.create_left_bv(X.getSizes()[-1])
    error_vec = [
        pytest_utils.compute_error_bv(
            comm, actions_petsc[i], X, Y, actions_python[i], Xpython
        )
        for i in range(len(actions_petsc))
    ]
    error = np.linalg.norm(error_vec)
    X.destroy()
    Y.destroy()
    linop.destroy()
    assert error < 1e-8
