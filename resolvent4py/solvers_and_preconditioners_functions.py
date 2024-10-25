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
    # Mat = pc.getFactorMatrix()
    # Mat.setMumpsIcntl(14,1000)
    # Mat.setMumpsIcntl(23,5000)
    # Mat.setMumpsIcntl(28,2)                 # Parallel permutations
    # opts = PETSc.Options()
    # opts["mat_mumps_icntl_24"] = 1      # Enable null pivot detection
    # opts["mat_mumps_cntl_4"] = 1e-6     # Threshold for static pivoting
    # opts["mat_mumps_cntl_5"] = 1e20     # Fixation for null pivots
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
    icntl14 = Mat.getMumpsIcntl(14)
    icntl23 = Mat.getMumpsIcntl(23)
    petscprint(comm, "Icntls = %d, %d"%(icntl14, icntl23))
    if info_g1 != 0:
        raise ValueError(
            f"MUMPS factorization failed with INFO(1) = {info_g1}  "
            f"and INFO(2) = {info_g2}'"
        )

