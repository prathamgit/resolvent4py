from petsc4py import PETSc
import scipy as sp
import numpy as np
import resolvent4py as res4py


def test_eigendecomposition(comm, square_random_matrix):
    """Test eigendecomp with different matrix sizes based on test level."""

    Apetsc, Apython = square_random_matrix
    omega = 20.0  # Angular frequency parameter
    r = 10  # Number of eigenvalues to compute

    Id = PETSc.Mat().createConstantDiagonal(Apetsc.getSizes(), 1.0, comm=comm)
    Id.scale(1j * omega)
    Id.convert(PETSc.Mat.Type.MPIAIJ)
    Id.axpy(-1.0, Apetsc)
    ksp = res4py.create_mumps_solver(comm, Id)
    linop = res4py.linear_operators.MatrixLinearOperator(comm, Id, ksp)
    krylov_dim = linop.get_dimensions()[0][-1] - 1
    r = np.min([r, krylov_dim - 3])
    lambda_fun = lambda x: 1j * omega - 1 / x
    D, _ = res4py.linalg.eig(linop, linop.solve, krylov_dim, r, lambda_fun)
    D = np.diag(D)

    ev, _ = sp.linalg.eig(Apython)
    ev_sorted = [ev[np.argmin(np.abs(ev - D[i]))] for i in range(len(D))]
    ev_sorted = np.asarray(ev_sorted)
    error = 100 * np.max(np.abs(ev_sorted - D) / np.abs(ev_sorted))

    assert error < 1e-1  # Max percent error < 1e-1
