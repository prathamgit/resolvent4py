import scipy as sp
import numpy as np
import resolvent4py as res4py
from petsc4py import PETSc
from .. import pytest_utils


def test_matrix_on_vectors(comm, square_random_matrix):
    r"""Test MatrixLinearOperator on vectors"""
    Apetsc, Apython = square_random_matrix
    ksp = res4py.create_mumps_solver(Apetsc)
    res4py.check_lu_factorization(Apetsc, ksp)
    linop = res4py.linear_operators.MatrixLinearOperator(Apetsc, ksp)
    x, xpython = pytest_utils.generate_random_vector(comm, Apython.shape[-1])

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
    assert error < 1e-10


def test_matrix_on_bvs(comm, square_random_matrix):
    r"""Test MatrixLinearOperator on BVs"""
    Apetsc, Apython = square_random_matrix
    ksp = res4py.create_mumps_solver(Apetsc)
    linop = res4py.linear_operators.MatrixLinearOperator(Apetsc, ksp)
    X, Xpython = pytest_utils.generate_random_bv(comm, (Apython.shape[0], 5))

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
    assert error < 1e-10
