import pytest
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import resolvent4py as res4py
from mpi4py import MPI
from petsc4py import PETSc
import os

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.sans-serif": ["Computer Modern"],
        "font.size": 18,
        "text.usetex": True,
    }
)

def L_generator(omega, A):
    comm = MPI.COMM_WORLD
    Rinv = res4py.create_AIJ_identity(comm, A.getSizes())
    Rinv.scale(1j * omega)
    Rinv.axpy(-1.0, A)
    ksp = res4py.create_mumps_solver(comm, Rinv)
    L = res4py.MatrixLinearOperator(comm, Rinv, ksp)
    return (L, L.solve_mat, (L.destroy,))

@pytest.fixture(scope="module")
def comm():
    return MPI.COMM_WORLD

@pytest.fixture(scope="module")
def truncation_data(tmp_path_factory, comm):
    path = tmp_path_factory.mktemp("balanced_truncation")
    N, Nb, Nc = 3, 2, 1
    complex_A = False
    fnames_jac = [str(path / "rows.dat"), str(path / "cols.dat"), str(path / "vals.dat")]
    if comm.Get_rank() == 0:
        os.makedirs(path) if os.path.isdir(path) == False else None
        A = np.asarray([[-0.1, 0.5, 0.0], [-0.5, -0.1, 0.0], [0, 0, -1]], \
                       dtype=np.complex128)
        if complex_A:
            A += 1j*np.eye(A.shape[0])
        evals, _ = sp.linalg.eig(A)
        print(evals)
        A = sp.sparse.coo_matrix(A)
        arrays = [A.row, A.col, A.data]
        for i, array in enumerate(arrays):
            vec = PETSc.Vec().createWithArray(
                array, len(array), None, MPI.COMM_SELF
            )
            res4py.write_to_file(MPI.COMM_SELF, fnames_jac[i], vec)
            vec.destroy()
        A = A.todense()
        B = np.asarray([[1, 0], [0, 0], [0, 1]])
        C = np.ones((3, 1))
        X = sp.linalg.solve_continuous_lyapunov(A, -B@B.conj().T)
        Y = sp.linalg.solve_continuous_lyapunov(A.conj().T, -C@C.conj().T)
        
        Sig_sq, Phi = sp.linalg.eig(X@Y)
        Psi = sp.linalg.inv(Phi).conj().T
        hankel = np.sqrt(Sig_sq)

        B = PETSc.Mat().createDense((N, Nb), None, B, comm=MPI.COMM_SELF)
        C = PETSc.Mat().createDense((N, Nc), None, C, comm=MPI.COMM_SELF)
        res4py.write_to_file(MPI.COMM_SELF, str(path / 'B.dat'), B)
        res4py.write_to_file(MPI.COMM_SELF, str(path / 'C.dat'), C)

    Nl = res4py.compute_local_size(N)
    A = res4py.read_coo_matrix(comm, fnames_jac, ((Nl, N), (Nl, N)))
    B = res4py.read_bv(comm, str(path / 'B.dat'), ((Nl, N), Nb))
    C = res4py.read_bv(comm, str(path / 'C.dat'), ((Nl, N), Nc))

    omegas, wlgs = [], []
    domega = 0.1
    intervals = np.arange(0, 31*domega, domega)
    idx = len(intervals)
    domega *= 10
    intervals = np.concatenate((intervals, np.arange(intervals[-1] + domega, \
                                                    intervals[-1] + 250*domega, \
                                                    domega)))
    poly_ords = 10*np.ones(len(intervals) - 1, dtype=np.int32)
    poly_ords[:idx] = 10

    for j in range(len(poly_ords)):
        points, wlg_j = np.polynomial.legendre.leggauss(poly_ords[j])
        of, oi = intervals[[j+1, j]]
        omegas_j = (of - oi)/2*points + (of + oi)/2
        omegas.extend(omegas_j)
        wlg_j *= 0.5*(of - oi)
        wlgs.extend(wlg_j)

    omegas = np.asarray(omegas)
    weights = np.asarray(wlgs) / np.pi

    if complex_A:
        omegas = np.concatenate((-np.flipud(omegas), omegas))
        weights = np.concatenate((np.flipud(weights), weights)) / 2

    L_gen = partial(L_generator, A=A)
    L_generators = [L_gen for _ in range(len(omegas))]

    return {
        'path': path,
        'N': N,
        'Nb': Nb,
        'Nc': Nc,
        'complex_A': complex_A,
        'fnames_jac': fnames_jac,
        'A': A,
        'B': B,
        'C': C,
        'omegas': omegas,
        'weights': weights,
        'L_generators': L_generators,
        'hankel': hankel,
    }

def test_balanced_truncation_files_exist(comm, truncation_data):
    # Example assertion: check file paths end with .dat
    for fname in truncation_data['fnames_jac']:
        assert fname.endswith('.dat')
    # Add more test logic as needed

def test_balanced_truncation_gramian_factors(comm, truncation_data):
    res4py.petscprint(comm, "Computing Gramian factors...")
    X, Y = res4py.model_reduction.compute_gramian_factors(truncation_data['L_generators'], \
                                                          truncation_data['omegas'], truncation_data['weights'], truncation_data['B'], truncation_data['C'])
    assert X is not None
    assert Y is not None

def test_balanced_truncation_balanced_projection(comm, truncation_data):
    res4py.petscprint(comm, "Computing balanced projection...")
    Phi, Psi, S = res4py.model_reduction.compute_balanced_projection(X, Y, truncation_data['N'])
    assert Phi is not None
    assert Psi is not None
    assert S is not None

def test_balanced_truncation_error(comm, truncation_data):
    error = 100*np.linalg.norm(np.diag(S) - truncation_data['hankel'])/np.linalg.norm(truncation_data['hankel'])
    res4py.petscprint(comm, "Percent error = %1.5e"%error)
    assert error < 1e-5
