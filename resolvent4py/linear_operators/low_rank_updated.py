from petsc4py import PETSc
from .linear_operator import LinearOperator
from ..linalg import compute_dense_inverse
from ..linalg import hermitian_transpose
from ..miscellaneous import create_dense_matrix
from .low_rank import LowRankLinearOperator

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
            XS = X.matMult(self.L.Sigma)
            VHT = hermitian_transpose(comm, self.L.V)
            M = VHT.matMult(XS)
            Id = PETSc.Mat().createConstantDiagonal(M.getSizes(), 1.0, comm)
            M.axpy(1.0, Id, structure=PETSc.Mat.Structure.SAME)
            Minv = compute_dense_inverse(comm,M)
            D = self.L.Sigma.matMult(Minv)
            mats = [XS, VHT, M, Minv, Id]
            for mat in mats: mat.destroy()
            W = LowRankLinearOperator(comm, X, D, Y, nblocks)
        except:
            W = None
        return W
    
    def create_intermediate_vectors(self):
        self.Ax = self.A.create_left_vector()
        self.ATx = self.A.create_right_vector()

    def create_intermediate_matrices_apply(self, sizes):
        AX = create_dense_matrix(self._comm, (self._dimensions[0], sizes[-1]))
        VTX, SX = self.L.create_intermediate_matrices(sizes)
        return (AX, VTX, SX)

    def create_intermediate_matrices_hermitian_transpose_apply(self, sizes):
        ATX = create_dense_matrix(self._comm,(self._dimensions[-1], sizes[-1]))
        UTX, STX = self.L.create_intermediate_matrices_hermitian_transpose(\
            sizes)
        return (ATX, UTX, STX)
    
    def create_intermediate_matrices_solve(self, sizes):
        AX = create_dense_matrix(self._comm, (self._dimensions[0], sizes[-1]))
        VTX, SX = self.W.create_intermediate_matrices(sizes)
        return (AX, VTX, SX)

    def create_intermediate_matrices_hermitian_transpose_solve(self, sizes):
        ATX = create_dense_matrix(self._comm,(self._dimensions[-1], sizes[-1]))
        UTX, STX = self.W.create_intermediate_matrices_hermitian_transpose(\
            sizes)
        return (ATX, UTX, STX)
    
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
    
    def apply_mat(self, X, Y=None, intermediate_mats=None):
        if intermediate_mats == None:
            destroy = True
            AX, VTX, SX = self.create_intermediate_matrices_apply(X.getSizes())
        else:
            destroy = False
            AX, VTX, SX = intermediate_mats
        AX = self.A.apply_mat(X, AX)
        Y = self.L.apply_mat(X, Y, (VTX, SX))
        Y.axpy(1.0, AX)
        if destroy:
            AX.destroy()
            VTX.destroy()
            SX.destroy()
        return Y
    
    def apply_hermitian_transpose_mat(self, X, Y=None, intermediate_mats=None):
        if intermediate_mats == None:
            destroy = True
            ATX, UTX, STX = \
                self.create_intermediate_matrices_hermitian_transpose_apply(\
                    X.getSizes())
        else:
            destroy = False
            ATX, UTX, STX = intermediate_mats
        ATX = self.A.apply_hermitian_transpose_mat(X, ATX)
        Y = self.L.apply_hermitian_transpose_mat(X, Y, (UTX, STX))
        Y.axpy(1.0, ATX)
        if destroy:
            ATX.destroy()
            UTX.destroy()
            STX.destroy()
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
    
    def solve_mat(self, X, Y=None, intermediate_mats=None):
        if intermediate_mats == None:
            destroy = True
            AX, VTX, SX = self.create_intermediate_matrices_solve(X.getSizes())
        else:
            destroy = False
            AX, VTX, SX = intermediate_mats
        AX = self.A.solve_mat(X, AX)
        Y = self.W.apply_mat(X, Y, (VTX, SX))
        Y.scale(-1.0)
        Y.axpy(1.0, AX)
        if destroy:
            AX.destroy()
            VTX.destroy()
            SX.destroy()
        return Y
    
    def solve_hermitian_transpose_mat(self, X, Y=None, intermediate_mats=None):
        if intermediate_mats == None:
            destroy = True
            ATX, UTX, STX = \
                self.create_intermediate_matrices_hermitian_transpose_solve(\
                    X.getSizes())
        else:
            destroy = False
            ATX, UTX, STX = intermediate_mats
        self.A.solve_hermitian_transpose_mat(X, ATX)
        Y = self.W.apply_hermitian_transpose_mat(X, Y, (UTX, STX))
        Y.scale(-1.0)
        Y.axpy(1.0, ATX)
        if destroy:
            ATX.destroy()
            UTX.destroy()
            STX.destroy()
        return Y
    
    def destroy_woodbury_operator(self):
        self.W.destroy()

    def destroy_low_rank_update(self):
        self.L.destroy()
        
    def destroy(self):
        self.destroy_woodbury_operator()
        self.destroy_low_rank_update()
        
