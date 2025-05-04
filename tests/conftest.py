import sys
import os
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
if rank != 0:
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

import pytest
from petsc4py import PETSc
import resolvent4py as res4py

@pytest.fixture(scope="session")
def comm():
    """MPI communicator fixture."""
    return MPI.COMM_WORLD


@pytest.fixture(scope="session")
def rank_size(comm):
    """Return rank and size of MPI communicator."""
    return comm.Get_rank(), comm.Get_size()


@pytest.fixture(
    params=[
        pytest.param(50, marks=pytest.mark.local),
        pytest.param(100, marks=pytest.mark.development),
        pytest.param(300, marks=pytest.mark.main),
    ]
)
def square_matrix_size(request):
    """Matrix size fixture with different sizes for different test levels."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param((50, 5), marks=pytest.mark.local),
        pytest.param((100, 10), marks=pytest.mark.development),
        pytest.param((300, 20), marks=pytest.mark.main),
    ]
)
def rectangular_matrix_size(request):
    """Matrix size fixture with different sizes for different test levels."""
    return request.param

@pytest.fixture
def square_random_matrix(comm, square_matrix_size):
    """Generate random test matrix."""

    N = square_matrix_size
    Nl = res4py.compute_local_size(N)
    
    Apetsc = res4py.generate_random_petsc_sparse_matrix(\
            comm, ((Nl, N), (Nl, N)), int(0.3*N**2), True)
    Adense = Apetsc.copy()
    Adense.convert(PETSc.Mat.Type.DENSE)
    Adense_seq = res4py.distributed_to_sequential_matrix(comm, Adense)
    Apython = Adense_seq.getDenseArray().copy()

    return Apetsc, Apython

@pytest.fixture
def rectangular_random_matrix(comm, rectangular_matrix_size):
    """Generate random test matrix."""

    Nr, Nc = rectangular_matrix_size
    Nrl = res4py.compute_local_size(Nr)
    Ncl = res4py.compute_local_size(Nc)
    
    Apetsc = res4py.generate_random_petsc_sparse_matrix(\
            comm, ((Nrl, Nr), (Ncl, Nc)), 0.1*Nr*Nc, False)
    Adense = Apetsc.copy()
    Adense.convert(PETSc.Mat.Type.DENSE)
    Adense_seq = res4py.distributed_to_sequential_matrix(comm, Adense)
    Apython = Adense_seq.getDenseArray().copy()

    return Apetsc, Apython

@pytest.fixture
def test_output_dir(tmp_path):
    """Create and return a temporary directory for test outputs."""
    return tmp_path / "test_output"
