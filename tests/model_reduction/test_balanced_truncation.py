from functools import partial
import numpy as np
import scipy as sp
import resolvent4py as res4py
from mpi4py import MPI

from .. import pytest_utils

def L_generator(omega, A):
    comm = MPI.COMM_WORLD
    Rinv = res4py.create_AIJ_identity(comm, A.getSizes())
    Rinv.scale(1j * omega)
    Rinv.axpy(-1.0, A)
    ksp = res4py.create_mumps_solver(comm, Rinv)
    L = res4py.linear_operators.MatrixLinearOperator(comm, Rinv, ksp)
    return (L, L.solve_mat, (L.destroy,))

@pytest.fixture(scope="module")
def comm():
    return MPI.COMM_WORLD

def test_balanced_truncation_real(comm):
    r"""Test balanced trunction."""
    complex = False
    N, rb, rc = 5, 3, 2
    Apetsc, Apython = pytest_utils.generate_random_matrix(comm, (N, N), complex)
    Bpetsc, Bpython = pytest_utils.generate_random_bv(comm, (N, rb), complex)
    Cpetsc, Cpython = pytest_utils.generate_random_bv(comm, (N, rc), complex)

    # Compute quadrature points and quadrature weights
    omegas, wlgs = [], []
    domega = 0.1
    intervals = np.arange(0, 31 * domega, domega)
    idx = len(intervals)
    domega *= 10
    intervals = np.concatenate(
        (
            intervals,
            np.arange(
                intervals[-1] + domega, intervals[-1] + 50 * domega, domega
            ),
        )
    )
    poly_ords = 5 * np.ones(len(intervals) - 1, dtype=np.int32)
    poly_ords[:idx] = 10

    for j in range(len(poly_ords)):
        points, wlg_j = np.polynomial.legendre.leggauss(poly_ords[j])
        of, oi = intervals[[j + 1, j]]
        omegas_j = (of - oi) / 2 * points + (of + oi) / 2
        omegas.extend(omegas_j)
        wlg_j *= 0.5 * (of - oi)
        wlgs.extend(wlg_j)

    omegas = np.asarray(omegas)
    weights = np.asarray(wlgs) / np.pi

    if complex:
        omegas = np.concatenate((-np.flipud(omegas), omegas))
        weights = np.concatenate((np.flipud(weights), weights)) / 2

    L_gen = partial(L_generator, A=Apetsc)
    L_generators = [L_gen for _ in range(len(omegas))]

    X, Y = res4py.model_reduction.compute_gramian_factors(
        L_generators, omegas, weights, Bpetsc, Cpetsc
    )
    r = 1
    Phi, Psi, S = res4py.model_reduction.compute_balanced_projection(X, Y, r)
    linop = res4py.linear_operators.MatrixLinearOperator(comm, Apetsc)
    Ar, _, _ = res4py.model_reduction.assemble_reduced_order_tensors(
        linop, Bpetsc, Cpetsc, Phi, Psi
    )
    linop.destroy()

    # Compute exact balanced truncation using scipy
    Qb = -Bpython @ Bpython.conj().T
    Qc = -Cpython @ Cpython.conj().T
    X = sp.linalg.solve_continuous_lyapunov(Apython, Qb)
    Y = sp.linalg.solve_continuous_lyapunov(Apython.conj().T, Qc)
    Hankel, Phi_python = sp.linalg.eig(X @ Y)
    Hankel = np.sqrt(Hankel)
    Psi_python = sp.linalg.inv(Phi_python).conj().T
    Phi_python = Phi_python[:, :r]
    Psi_python = Psi_python[:, :r]
    Ar_python = Psi_python.conj().T@Apython@Phi_python

    error = 100 * np.linalg.norm(np.diag(S)[0] - Hankel[0]) / \
        np.linalg.norm(Hankel[0])
    assert error < 2
    error = 100*np.linalg.norm(Ar - Ar_python) / np.linalg.norm(Ar_python)
    assert error < 2
