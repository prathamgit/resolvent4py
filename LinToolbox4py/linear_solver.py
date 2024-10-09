import scipy as sp
import numpy as np
from petsc4py import PETSc
from mpi4py import MPI
from .petsc4py_helper_functions import mat_solve_hermitian_transpose, compute_dense_inverse

class LinearSolver:
    r"""
        This class creates an instance of a solver to compute :math:`y = H x`, where

        .. math::
            H = \mathbb{P}\left(A + U K V^*\right)^{-1}\mathbb{P}

        is the inverse of :math:`L` on the range of the projection :math:`\mathbb{P}`.
        See :meth:`LinToolbox4py.linear_operator.LinearOperator` for a description of 
        :math:`L`, :math:`\mathbb{P}`, :math:`U`, :math:`V` and :math:`K`.
        The matrix :math:`R = \left(A + U K V^*\right)^{-1}` is computed using the Woodbury matrix
        identity

        .. math::
            R = A^{-1} - X D Y^{*},

        where :math:`X = A^{-1}U`, :math:`Y = A^{-*}V` and :math:`D = K\left(I + V^*A^{-1}U K\right)^{-1}`.
        

        :param lin_op: an instance of the :meth:`|pkgname|.linear_operator.LinearOperator` class.
        :param solver_type: type of solver (currently only 'mumps' is supported)
    """

    def __init__(
        self,
        lin_op,
        solver_type='preonly'
    ):
        self.lin_op = lin_op
        self.solver_type = solver_type
        self.create_solver()
        self.set_woodbury_factors()

    
    def create_solver(
        self
    ):
        r"""
        Compute the LU decomposition of :math:`A` using MUMPS

        :return: an instance of the KSP structure
        :rtype: PETSc.KSP
        """

        self.ksp = PETSc.KSP().create(comm=self.lin_op.comm)
        self.ksp.setOperators(self.lin_op.A)
        self.ksp.setType(self.solver_type)

        if self.solver_type == 'preonly' and self.lin_op.comm.Get_size()>1:
            pc = self.ksp.getPC()
            pc.setType('lu')
            pc.setFactorSolverType('mumps')
            pc.setUp()

        self.ksp.setUp()

    def set_woodbury_factors(
        self,
        factors=None
    ):
        if factors == None:
            self.X = self.lin_op.U.duplicate()
            self.ksp.matSolve(self.lin_op.U,self.X)
            self.Y = mat_solve_hermitian_transpose(self,self.lin_op.V)

            self.lin_op.V.hermitianTranspose()
            M = self.lin_op.V.matMult(self.X).matMult(self.lin_op.K)
            Id = PETSc.Mat().createConstantDiagonal(M.getSizes(),1.0,comm=self.lin_op.comm)
            M.axpy(1.0,Id,structure=PETSc.Mat.Structure.SAME)
            self.D = self.lin_op.K.matMult(compute_dense_inverse(self.lin_op.comm,M))
            self.lin_op.V.hermitianTranspose()
        else:
            self.X, self.D, self.Y = factors

    def apply_woodbury_factors(
        self,
        x,
        mode='direct'
    ):
        y = x.duplicate()

        if mode == 'direct':
            z, Dz = self.D.createVecs()
            self.Y.multHermitian(x,z)
            self.D.mult(z,Dz)
            self.X.mult(Dz,y)
        elif mode == 'adjoint':
            Dz, z = self.D.createVecs()
            self.X.multHermitian(x,z)
            self.D.multHermitian(z,Dz)
            self.Y.mult(Dz,y)

        return y

    def solve(
            self,
            x,
            mode='direct'
    ):
        r"""
            Computes :math:`y = Hx` if mode = 'direct' or `y = H^*x` if mode = 'adjoint'. 

            :param x: a vector of size :math:`N`
            :type x: PETSc.Vec
            :param mode: one of 'direct' or 'adjoint'
            :type mode: str

            :return: a vector of size :math:`N`
            :rtype: PETSc.Vec
        """

        Px = self.lin_op.apply_projection(x,mode)
        z = Px.duplicate()
        if mode == 'direct':
            self.ksp.solve(Px,z)
        elif mode == 'adjoint':
            Px.conjugate()
            self.ksp.solveTranspose(Px,z)
            Px.conjugate()
            z.conjugate()
        z.axpy(-1.0,self.apply_woodbury_factors(Px,mode))
        y = self.lin_op.apply_projection(z,mode)

        return y