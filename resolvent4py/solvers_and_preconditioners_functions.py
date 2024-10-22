from petsc4py import PETSc

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
    pc = ksp.getPC()
    pc.setType('lu')
    pc.setFactorSolverType('mumps')
    pc.setUp()
    ksp.setUp()

    return ksp