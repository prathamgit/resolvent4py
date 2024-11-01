from petsc4py import PETSc
from .linear_operator import LinearOperator
from ..linalg import compute_dense_inverse
from ..linalg import mat_solve_hermitian_transpose
from ..linalg import hermitian_transpose
from ..miscellaneous import create_dense_matrix
from .low_rank import LowRankLinearOperator

# class LowRankUpdatedLinearOperator(LinearOperator):
#     r"""
#         Class for a linear operator of the form

#         .. math::

#             L = A + B K C^*
        
#         where :math:`A` is an instance of any subclass of the 
#         `resolvent4py.linear_operator.LinearOperator` class,
#         and :math:`B`, :math:`K` and :math:`C` are low-rank (dense) matrices of 
#         conformal size.
#         If :code:`A.solve()` is enabled, then the :code:`solve()` method 
#         in this class if implemented using the Woodbury matrix identity

#         .. math::

#             L^{-1} = A^{-1} - X D Y^*,

#         where

#         .. math::

#             \textcolor{black}{X}
#             = A^{-1}B,\quad \textcolor{black}{D} = 
#             K\left(I + C^* A^{-1}B K\right)^{-1},\quad
#             \textcolor{black}{Y} = A^{-*}C,

#         are the Woodbury factors.
#         The same goes for the :code:`solve_hermitian_transpose()` method. 

#         :param comm: MPI communicator (one of :code:`MPI.COMM_WORLD` or
#             :code:`MPI.COMM_SELF`)
#         :param A: instance of a subclass of the 
#             `resolvent4py.linear_operators.LinearOperator` class
#         :param B: dense PETSc matrix
#         :type B: PETSc.Mat.Type.DENSE
#         :param K: dense PETSc matrix
#         :type K: PETSc.Mat.Type.DENSE
#         :param C: dense PETSc matrix
#         :type C: PETSc.Mat.Type.DENSE
#         :param woodbury_factors: [optional] 3-tuple of PETsc dense matrices
#             containing the Woodbury factors :code:`X`, :code:`D` and :code:`Y`
#             in that order.
#             If :code:`woodbury_factors == None` and :code:`A.solve()` is
#             enabled, the factors are computed using the 
#             :code:`compute_woodbury_factors()` method.
#         :param nblocks: [optional] number of blocks (if the linear operator \
#             has block structure)
#         :type nblocks: int
#     """

#     def __init__(self, comm, A, B, K, C, woodbury_factors=None, nblocks=None):

#         self.A = A
#         self.B = B
#         self.K = K
#         self.C = C
#         if woodbury_factors == None:
#             try:
#                 self.compute_woodbury_factors(comm)
#             except:
#                 self.X, self.D, self.Y = None, None, None
#         else:
#             self.X, self.D, self.Y = woodbury_factors

#         super().__init__(comm, 'LowRankUpdatedLinearOperator', \
#                          A.get_dimensions(), nblocks)
            

#     def compute_woodbury_factors(self, comm):
#         r"""
#             Compute Woodbury factors :code:`X`, :code:`D` and :code:`Y` 
#             and set corresponding attributes.

#             :rtype: None
#         """
#         self.X = self.B.duplicate()
#         self.A.ksp.matSolve(self.B, self.X)
#         self.Y = mat_solve_hermitian_transpose(self.A.ksp, self.C)
#         XK = self.X.matMult(self.K)
#         CHT = hermitian_transpose(comm, self.C)
#         M = CHT.matMult(XK)
#         Id = PETSc.Mat().createConstantDiagonal(M.getSizes(),1.0,comm=comm)
#         M.axpy(1.0,Id,structure=PETSc.Mat.Structure.SAME)
#         Minv = compute_dense_inverse(comm,M)
#         self.D = self.K.matMult(Minv)
#         Minv.destroy()
#         M.destroy()
#         XK.destroy()
#         CHT.destroy()
#         Id.destroy()



#     def apply_low_rank_factors(self, F1, F2, F3, x):
#         r"""
#             :param F1: a dense PETSc matrix
#             :type F1: PETSc.Mat.Type.DENSE
#             :param F2: a dense PETSc matrix
#             :type F2: PETSc.Mat.Type.DENSE
#             :param F3: a dense PETSc matrix
#             :type F3: PETSc.Mat.Type.DENSE
#             :param x: a PETSc vector
#             :type x: PETSc.Vec.Type.STANDARD

#             :return: :math:`F_1 F_2 F_3^* x`
#             :rtype: PETSc.Vec.Type.STANDARD
#         """
#         z = F3.createVecRight()
#         q = F2.createVecLeft()
#         y = F1.createVecLeft()
#         F3.multHermitian(x,z)
#         F2.mult(z,q)
#         F1.mult(q,y)
#         return y
    
#     def apply_low_rank_factors_mat(self, F1, F2, F3, X, Y=None):
#         r"""
#             :param F1: a dense PETSc matrix
#             :type F1: PETSc.Mat.Type.DENSE
#             :param F2: a dense PETSc matrix
#             :type F2: PETSc.Mat.Type.DENSE
#             :param F3: a dense PETSc matrix
#             :type F3: PETSc.Mat.Type.DENSE
#             :param X: a PETSc matrix
#             :type X: PETSc.Mat.Type.DENSE

#             :return: :math:`F_1 F_2 F_3^* X`
#             :rtype: PETSc.Mat.Type.DENSE
#         """
#         colsizes = X.getSizes()[-1]
#         Q = create_dense_matrix(self.get_comm(), (F3.getSizes()[-1], colsizes))
#         Z = create_dense_matrix(self.get_comm(), (F2.getSizes()[0], colsizes))
#         Y = create_dense_matrix(self.get_comm(), (F1.getSizes()[0], colsizes)) \
#             if Y == None else Y
        
#         F3.hermitianTranspose()
#         F3.matMult(X, Q)
#         F3.hermitianTranspose()
#         F2.matMult(Q, Z)
#         F1.matMult(Z, Y)
#         Q.destroy()
#         Z.destroy()
#         return Y
    
#     def apply_low_rank_factors_hermitian_transpose(self, F1, F2, F3, x):
#         r"""
#             :param F1: a dense PETSc matrix
#             :type F1: PETSc.Mat.Type.DENSE
#             :param F2: a dense PETSc matrix
#             :type F2: PETSc.Mat.Type.DENSE
#             :param F3: a dense PETSc matrix
#             :type F3: PETSc.Mat.Type.DENSE
#             :param x: a PETSc vector
#             :type x: PETSc.Vec.Type.STANDARD

#             :return: :math:`F_3 F_2^* F_1^* x`
#             :rtype: PETSc.Vec.Type.STANDARD
#         """
#         z = F1.createVecRight()
#         q = F2.createVecRight()
#         y = F3.createVecLeft()
#         F1.multHermitian(x,z)
#         F2.multHermitian(z,q)
#         F3.mult(q,y)
#         return y
    
#     def apply_low_rank_factors_hermitian_transpose_mat(self, F1, F2, F3, X, \
#                                                        Y=None):
#         r"""
#             :param F1: a dense PETSc matrix
#             :type F1: PETSc.Mat.Type.DENSE
#             :param F2: a dense PETSc matrix
#             :type F2: PETSc.Mat.Type.DENSE
#             :param F3: a dense PETSc matrix
#             :type F3: PETSc.Mat.Type.DENSE
#             :param X: a PETSc matrix
#             :type X: PETSc.Mat.Type.DENSE

#             :return: :math:`F_3 F_2^* F_1^* X`
#             :rtype: PETSc.Mat.Type.DENSE
#         """
#         colsizes = X.getSizes()[-1]
#         Q = create_dense_matrix(self.get_comm(), (F1.getSizes()[-1], colsizes))
#         Z = create_dense_matrix(self.get_comm(), (F2.getSizes()[-1], colsizes))
#         Y = create_dense_matrix(self.get_comm(), (F3.getSizes()[0], colsizes)) \
#             if Y == None else Y
#         F1.hermitianTranspose()
#         F1.matMult(X, Q)
#         F1.hermitianTranspose()
#         F2.hermitianTranspose()
#         F2.matMult(Q, Z)
#         F2.hermitianTranspose()
#         F3.matMult(Z, Y)
#         Q.destroy()
#         Z.destroy()
#         return Y

#     def apply(self, x, y=None):
#         y = self.A.apply(x, y)
#         z = self.apply_low_rank_factors(self.B, self.K, self.C, x)
#         y.axpy(1.0,z)
#         z.destroy()
#         return y
    
#     def apply_mat(self, X, Y=None):
#         Y = self.A.apply_mat(X, Y)
#         Z = self.apply_low_rank_factors_mat(self.B, self.K, self.C, X)
#         Y.axpy(1.0, Z)
#         Z.destroy()
#         return Y
    
#     def apply_hermitian_transpose(self, x, y=None):
#         y = self.A.apply_hermitian_transpose(x, y)
#         z = self.apply_low_rank_factors_hermitian_transpose(self.B, self.K,\
#                                                             self.C, x)
#         y.axpy(1.0,z)
#         z.destroy()
#         return y
    
#     def apply_hermitian_transpose_mat(self, X, Y=None):
#         Y = self.A.apply_hermitian_transpose_mat(X, Y)
#         Z = self.apply_low_rank_factors_hermitian_transpose_mat(self.B, self.K,\
#                                                                 self.C, X)
#         Y.axpy(1.0,Z)
#         Z.destroy()
#         return Y
    
#     def solve(self, x, y=None):
#         y = self.A.solve(x, y)
#         z = self.apply_low_rank_factors(self.X,self.D,self.Y,x)
#         y.axpy(-1.0,z)
#         z.destroy()
#         return y
    
#     def solve_mat(self, X, Y=None):
#         Y = self.A.solve_mat(X, Y)
#         Z = self.apply_low_rank_factors_mat(self.X,self.D,self.Y,X)
#         Y.axpy(-1.0, Z)
#         Z.destroy()
#         return Y
        
#     def solve_hermitian_transpose(self, x, y=None):
#         y = self.A.solve_hermitian_transpose(x) if y == None else \
#             self.A.solve_hermitian_transpose(x, y)
#         z = self.apply_low_rank_factors_hermitian_transpose(self.X, self.D,
#                                                             self.Y, x)
#         y.axpy(-1.0, z)
#         z.destroy()
#         return y
    
#     def solve_hermitian_transpose_mat(self, X, Y=None):
#         Y = self.A.solve_hermitian_transpose_mat(X, Y)
#         Z = self.apply_low_rank_factors_hermitian_transpose_mat(self.X, self.D,
#                                                             self.Y, X)
#         Y.axpy(-1.0, Z)
#         Z.destroy()
#         return Y
    
#     def destroy_woodbury_factors(self):
#         if self.X != None:
#             self.X.destroy()
#             self.D.destroy()
#             self.Y.destroy()
    
#     def destroy(self):
#         self.destroy_woodbury_factors()
#         self.B.destroy()
#         self.K.destroy()
#         self.C.destroy()
        


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

        super().__init__(comm, 'LowRankUpdatedLinearOperator', \
                         A.get_dimensions(), nblocks)
        
        self.create_intermediate_vectors()
            

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
        self.Ax = self.create_left_vector()
        self.ATx = self.create_right_vector()

    def create_intermediate_matrices_apply(self, sizes):
        AX = create_dense_matrix(self._comm, (self._dimensions()[0], sizes[-1]))
        VTX, SX = self.L.create_intermediate_matrices(sizes)
        return (AX, VTX, SX)

    def create_intermediate_matrices_hermitian_transpose_apply(self, sizes):
        ATX = create_dense_matrix(self._comm,(self._dimensions()[-1],sizes[-1]))
        UTX, STX = self.L.create_intermediate_matrices_hermitian_transpose(\
            sizes)
        return (ATX, UTX, STX)
    
    def create_intermediate_matrices_solve(self, sizes):
        AX = create_dense_matrix(self._comm, (self._dimensions()[0], sizes[-1]))
        VTX, SX = self.W.create_intermediate_matrices(sizes)
        return (AX, VTX, SX)

    def create_intermediate_matrices_hermitian_transpose_solve(self, sizes):
        ATX = create_dense_matrix(self._comm,(self._dimensions()[-1],sizes[-1]))
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
        y.axpy(1.0, self.ATX)
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
                self.create_intermediate_matrices_hermitian_transpose_apply(\
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
        
