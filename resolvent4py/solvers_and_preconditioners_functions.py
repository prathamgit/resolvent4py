from petsc4py import PETSc
from .random import generate_random_petsc_vector
from .miscellaneous import petscprint

def create_mumps_solver(comm,A):
    r"""
        Compute an LU factorization of the matrix A using 
        `MUMPS <https://mumps-solver.org/index.php?page=doc>`

        :param comm: MPI communicator (:code:`MPI.COMM_WORLD` or 
            :code:`MPI.COMM_SELF`)
        :param A: PETSc matrix
        :type A: PETSc.Mat.Type.AIJ
        :return ksp: PETSc KSP solver
        :rtype ksp: `KSP`_
    """
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOperators(A)
    ksp.setType('preonly')
    ksp.setUp()
    pc = ksp.getPC()
    pc.setType('lu')
    pc.setFactorSolverType('mumps')
    pc.setUp()
    
    return ksp

def check_lu_factorization(comm, A, ksp):

    sizes = A.getSizes()[0]
    b = generate_random_petsc_vector(comm, sizes)
    x = b.duplicate()
    ksp.solve(b,x)
    pc = ksp.getPC()
    Mat = pc.getFactorMatrix()
    info_g1 = Mat.getMumpsInfog(1)
    info_g2 = Mat.getMumpsInfog(2)
    if info_g1 != 0:
        raise ValueError(
            f"MUMPS factorization failed with INFO(1) = {info_g1}  "
            f"and INFO(2) = {info_g2}'"
        )

