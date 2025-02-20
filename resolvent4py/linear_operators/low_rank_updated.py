from ..linalg import bv_add
from .low_rank import LowRankLinearOperator
from .linear_operator import LinearOperator

from .. import np
from .. import sp
from .. import MPI
from .. import PETSc
from .. import SLEPc


class LowRankUpdatedLinearOperator(LinearOperator):
    r"""
        Class for a linear operator of the form

        .. math::

            L = A + B K C^*
        
        where :math:`A` is an instance of any subclass of the 
        `resolvent4py.linear_operator.LinearOperator` class,
        and :math:`B`, :math:`K` and :math:`C` are low-rank (dense) matrices of 
        conformal size.
        If :code:`A.solve()` is enabled, then the :code:`solve()` method 
        in this class if implemented using the Woodbury matrix identity

        .. math::

            L^{-1} = A^{-1} - X D Y^*,

        where

        .. math::

            \textcolor{black}{X}
            = A^{-1}B,\quad \textcolor{black}{D} = 
            K\left(I + C^* A^{-1}B K\right)^{-1},\quad
            \textcolor{black}{Y} = A^{-*}C,

        are the Woodbury factors.
        The same goes for the :code:`solve_hermitian_transpose()` method. 

        :param comm: MPI communicator (one of :code:`MPI.COMM_WORLD` or
            :code:`MPI.COMM_SELF`)
        :param A: instance of a subclass of the 
            `resolvent4py.linear_operators.LinearOperator` class
        :param B: dense PETSc matrix
        :type B: PETSc.Mat.Type.DENSE
        :param K: dense PETSc matrix
        :type K: PETSc.Mat.Type.DENSE
        :param C: dense PETSc matrix
        :type C: PETSc.Mat.Type.DENSE
        :param woodbury_factors: [optional] 3-tuple of PETsc dense matrices
            containing the Woodbury factors :code:`X`, :code:`D` and :code:`Y`
            in that order.
            If :code:`woodbury_factors == None` and :code:`A.solve()` is
            enabled, the factors are computed using the 
            :code:`compute_woodbury_factors()` method.
        :param nblocks: [optional] number of blocks (if the linear operator \
            has block structure)
        :type nblocks: int
    """

    def __init__(self, comm, A, B, K, C, woodbury_factors=None, nblocks=None):

        self.A = A
        self.L = LowRankLinearOperator(comm, B, K, C, nblocks)
        self.W = self.compute_woodbury_operator(comm, nblocks) \
            if woodbury_factors == None else \
                LowRankLinearOperator(comm, *woodbury_factors, nblocks)
        self.create_intermediate_vectors()
        super().__init__(comm, 'LowRankUpdatedLinearOperator', \
                         A.get_dimensions(), nblocks)
        
    
    def compute_woodbury_operator(self, comm, nblocks):
        r"""
            :return: a :code:`LowRankLinearOperator` constructed from the 
                Woodbury factors :code:`X`, :code:`D` and :code:`Y` 
        """
        try:
            X = self.A.solve_mat(self.L.U)
            Y = self.A.solve_hermitian_transpose_mat(self.L.V)
            S = PETSc.Mat().createDense(self.L.Sigma.shape, None, \
                                        self.L.Sigma, MPI.COMM_SELF)
            XS = SLEPc.BV().create(comm)
            XS.setSizes(X.getSizes()[0], self.L.Sigma.shape[-1])
            XS.setType('mat')
            XS.mult(1.0, 0.0, X, S)
            M = XS.dot(self.L.V)
            Ma = M.getDenseArray()
            D = self.L.Sigma@sp.linalg.inv(np.eye(Ma.shape[0]) + Ma)
            W = LowRankLinearOperator(comm, X, D, Y, nblocks)
            XS.destroy()
            M.destroy()
            S.destroy()
        except:
            W = None
        return W
    
    def create_intermediate_vectors(self):
        self.Ax = self.A.create_left_vector()
        self.ATx = self.A.create_right_vector()

    def create_intermediate_bv(self, m):
        X = SLEPc.BV().create(self._comm)
        X.setSizes(self._dimensions[0], m)
        X.setType('mat')
        return X

    def create_intermediate_bv_hermitian_transpose(self, m):
        X = SLEPc.BV().create(self._comm)
        X.setSizes(self._dimensions[-1], m)
        X.setType('mat')
        return X

    def apply(self, x, y=None):
        self.Ax = self.A.apply(x, self.Ax)
        y = self.L.apply(x, y)
        y.axpy(1.0, self.Ax)
        return y
    
    def apply_hermitian_transpose(self, x, y=None):
        self.ATx = self.A.apply_hermitian_transpose(x, self.ATx)
        y = self.L.apply_hermitian_transpose(x, y)
        y.axpy(1.0, self.ATx)
        return y
    
    def apply_mat(self, X, Y=None, Z=None):
        destroy = False
        if Z == None:
            destroy = True
            Z = self.create_intermediate_bv(X.getSizes()[-1])
        Z = self.A.apply_mat(X, Z)
        Y = self.L.apply_mat(X, Y)
        bv_add(1.0, Y, Z)
        Z.destroy() if destroy else None
        return Y
    
    def apply_hermitian_transpose_mat(self, X, Y=None, Z=None):
        destroy = False
        if Z == None:
            destroy = True
            Z = self.create_intermediate_bv_hermitian_transpose(\
                X.getSizes()[-1])
        Z = self.A.apply_hermitian_transpose_mat(X, Z)
        Y = self.L.apply_hermitian_transpose_mat(X, Y)
        bv_add(1.0, Y, Z)
        Z.destroy() if destroy else None
        return Y
    
    def solve(self, x, y=None):
        self.Ax = self.A.solve(x, self.Ax)
        y = self.W.apply(x, y)
        y.scale(-1.0)
        y.axpy(1.0, self.Ax)
        return y
    
    def solve_hermitian_transpose(self, x, y=None):
        self.ATx = self.A.solve_hermitian_transpose(x, self.ATx)
        y = self.W.apply_hermitian_transpose(x, y)
        y.scale(-1.0)
        y.axpy(1.0, self.ATx)
        return y
    
    def solve_mat(self, X, Y=None, Z=None):
        destroy = False
        if Z == None:
            destroy = True
            Z = self.create_intermediate_bv(X.getSizes()[-1])
        Z = self.A.solve_mat(X, Z)
        Y = self.W.apply_mat(X, Y)
        Y.scale(-1.0)
        bv_add(1.0, Y, Z)
        Z.destroy() if destroy else None
        return Y
    
    def solve_hermitian_transpose_mat(self, X, Y=None, Z=None):
        destroy = False
        if Z == None:
            destroy = True
            Z = self.create_intermediate_bv_hermitian_transpose(\
                X.getSizes()[-1])
        Z = self.A.solve_hermitian_transpose_mat(X, Z)
        Y = self.W.apply_hermitian_transpose_mat(X, Y)
        Y.scale(-1.0)
        bv_add(1.0, Y, Z)
        Z.destroy() if destroy else None
        return Y
    
    def destroy_woodbury_operator(self):
        self.W.destroy() if self.W != None else None

    def destroy_low_rank_update(self):
        self.L.destroy()

    def destroy_intermediate_vectors(self):
        self.Ax.destroy()
        self.ATx.destroy()
        
    def destroy(self):
        self.destroy_intermediate_vectors()
        self.destroy_woodbury_operator()
        self.destroy_low_rank_update()