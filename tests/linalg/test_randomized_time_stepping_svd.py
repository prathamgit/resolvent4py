import pytest
import scipy as sp
import numpy as np
import resolvent4py as res4py
from mpi4py import MPI
from petsc4py import PETSc


def test_randomized_time_stepping_svd(
    comm, square_random_negative_semidefinite_matrix
):
    """Test randomized timestepping SVD with different matrix sizes based on test level."""
    s = 100000.0
    omega = np.arange(-2 * s, 2 * s, s)
    Apetsc, Apython = square_random_negative_semidefinite_matrix
    krylov_dim = np.min([30, np.min(Apython.shape)])
    linop = res4py.linear_operators.MatrixLinearOperator(comm, Apetsc)
    n_cycles = 10
    n_periods = 100
    n_timesteps = 200
    r = np.min([3, krylov_dim])
    _, S, _ = res4py.linalg.randomized_time_stepping_svd(
        linop, omega, n_periods, n_timesteps, krylov_dim, n_cycles, r
    )

    for i in range(len(omega)):
        w = omega[i]
        temp = np.linalg.inv(1j * w * np.eye(np.min(Apython.shape)) - Apython)
        _, s_act, _ = sp.linalg.svd(temp, full_matrices=False)
        print(w)
        print(s_act[:r])
        print(S[i, :])

        if w == s:
            error = 100 * np.max((np.diag(S[i, :]) - s_act[:r]) / s_act[:r])
            print(error)
            assert error < 5
