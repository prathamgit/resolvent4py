import scipy as sp
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
from .petsc4py_helper_functions import mat_solve_hermitian_transpose

class LinearSolver:
    r"""
        This class creates an instance of a solver to compute :math:`y = H x`, where
    
        .. math::
            H = \mathbb{P}\left(A + U K V^*\right)^{-1}\mathbb{P}.

        See :meth:`LinToolbox4py.linear_operator.LinearOperator` for a description of 
        :math:`\mathbb{P}`, :math:`U`, :math:`V` and :math:`K`.
        
        :param lin_op: an instance of the :meth:`LinToolbox4py.linear_operator.LinearOperator` class.
        :param solver_type: type of solver (currently only 'mumps' is supported)
    """

    def __init__(
        self,
        lin_op,
        solver_type='mumps'
    ):
        self.lin_op = lin_op
        self.solver_type = solver_type

        if solver_type == 'mumps':
            self.create_mumps_solver()


    def create_mumps_solver(
        self
    ):
        r"""
        Compute the LU decomposition of :math:`A` using MUMPS

        :return: an instance of the KSP structure
        :rtype: PETSc.KSP
        """

        self.ksp = PETSc.KSP().create()
        self.ksp.setOperators(self.lin_op.A)
        self.ksp.setType('preonly')
        pc = self.ksp.getPC()
        pc.setType('lu')
        pc.setFactorSolverType('mumps')
        pc.setUp()
        self.ksp.setUp()

    def create_woodbury_factors(
        self
    ):
        self.X = self.lin_op.U.duplicate()
        self.ksp.matSolve(self.lin_op.U,self.X)
        self.Y = mat_solve_hermitian_transpose(self.lin_op.V)

        self.lin_op.V.hermitianTranspose()
        F = self.lin_op.V.matMult(self.X)
        self.lin_op.V.hermitianTranspose()
        R = F.copy()
        R.convert(PETSc.Mat.Type.SEQDENSE)
        R_array = R.getDenseArray()
        K = self.lin_op.K.copy()
        K.convert(PETSc.Mat.Type.SEQDENSE)
        K_array = K.getDenseArray()
        M_array = K_array@sp.linalg.inv(np.eye(R_array.shape[0]) + R_array@K_array)
        M = PETSc.Mat().createDense(R.getSize(),None,M_array,MPI.COMM_SELF)
        M.convert(PETSc.Mat.Type.DENSE,out=F)
        self.R = F.copy()

    

    # def solve(
    #         self,
    #         x,
    #         mode='direct'
    # ):
    #     r"""
    #         Computes :math:`y = \mathbb{P}\left(A + U V^*\right)^{-1}\mathbb{P}x` if mode = 'direct' or 
    #         `y = \mathbb{P}^*\left(A + U V^*\right)^{-*}\mathbb{P}^*x` if mode = 'adjoint'. The action of
    #         :math:`\left(A + U V^*\right)^{-1}` on a vector is computed using the Woodbury matrix identity.

    #         :param x: a distributed vector of size :math:`N`
    #         :type x: PETSc.Vec
    #         :param mode: one of 'direct' and 'adjoint'
    #         :type mode: str

    #         :return: a distributed vector of size :math:`N`
    #         :rtype: PETSc.Vec
    #     """

    
        