from .linear_operator import LinearOperator
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from ..linalg import hermitian_transpose
from ..miscellaneous import create_dense_matrix

class LowRankLinearOperator(LinearOperator):
    r"""
        Class for a linear operator of the form

        .. math::

            L = U \Sigma V^*,

        where :math:`U`, :math:`\Sigma` and :math:`V` are matrices of
        conformal sizes (and :math:`\Sigma` is not necessarily diagonal).

        :param comm: MPI communicator (one of :code:`MPI.COMM_WORLD` or
            :code:`MPI.COMM_SELF`)
        :param U: dense PETSc matrix
        :type U: PETSc.Mat.Type.DENSE
        :param Sigma: dense PETSc matrix
        :type Sigma: PETSc.Mat.Type.DENSE
        :param V: dense PETSc matrix
        :type V: PETSc.Mat.Type.DENSE
        :param nblocks: [optional] number of blocks (if the linear operator \
            has block structure)
        :type nblocks: int
    """
    def __init__(self, comm, U, Sigma, V, nblocks=None):
        self.U = U
        self.Sigma = Sigma
        self.V = V
        dimensions = (U.getSizes()[0],V.getSizes()[0])
        super().__init__(comm, 'LowRankLinearOperator', dimensions, nblocks)

        self.create_intermediate_vectors()

    def create_intermediate_vectors(self):
        r"""
            Create intermediate vectors used during the computation of 
            matrix-vector products :math:`U \Sigma V^* x` and 
            :math:`V\Sigma^* U^* x`.
        """
        self.VTx = self.V.createVecRight()
        self.Sigmax = self.Sigma.createVecLeft()
        self.SigmaTx = self.Sigma.createVecRight()
        self.UTx = self.U.createVecRight()

    def create_intermediate_matrices(self, sizes):
        r"""
            Creates intermediate matrices
            used during the computation of matrix-matrix products 
            :math:`U \Sigma V^* X` 

            :param sizes: size of the matrix :math:`X`
            :type sizes: `MatSizeSpec`_
            :return: a 2-tuple of PETSc matrices of type 
                PETSc.Mat.Type.DENSE where the products :math:`V^* X` and 
                :math:`\Sigma V^* X` can be stored
        """
        sizes_VTX = (self.V.getSizes()[-1], sizes[-1])
        VTX = create_dense_matrix(self._comm, sizes_VTX)
        sizes_SX = (self.Sigma.getSizes()[0], sizes[-1])
        SigmaX = create_dense_matrix(self._comm, sizes_SX)
        return (VTX, SigmaX)

    def create_intermediate_matrices_hermitian_transpose(self, sizes):
        r"""
            Creates intermediate matrices
            used during the computation of matrix-matrix products 
            :math:`V \Sigma^* U^* X` 

            :param sizes: size of the matrix :math:`X`
            :type sizes: `MatSizeSpec`_
            :return: a 2-tuple of PETSc matrices of type 
                PETSc.Mat.Type.DENSE where the products :math:`U^* X` and 
                :math:`\Sigma^* U^* X` can be stored
        """
        sizes_UTX = (self.U.getSizes()[-1], sizes[-1])
        UTX = create_dense_matrix(self._comm, sizes_UTX)
        sizes_STX = (self.Sigma.getSizes()[-1], sizes[-1])
        SigmaTX = create_dense_matrix(self._comm, sizes_STX)
        return (UTX, SigmaTX)

    def apply(self, x, y=None):
        y = self.U.createVecLeft() if y == None else y
        self.V.multHermitian(x,self.VTx)
        self.Sigma.mult(self.VTx,self.Sigmax)
        self.U.mult(self.Sigmax,y)
        return y

    def apply_mat(self, X, Y=None, intermediate_mats=None):
        if intermediate_mats == None:
            destroy = True
            VTX, SX = self.create_intermediate_matrices(X.getSizes())
        else:
            destroy = False
            VTX, SX = intermediate_mats
        self.V.hermitianTranspose()
        self.V.matMult(X, VTX)
        self.Sigma.matMult(VTX, SX)
        Y = self.U.matMult(SX, Y)
        self.V.hermitianTranspose()
        if destroy:
            VTX.destroy()
            SX.destroy()
        return Y
    
    def apply_hermitian_transpose(self, x, y=None):
        y = self.V.createVecLeft() if y == None else y
        self.U.multHermitian(x,self.UTx)
        self.Sigma.multHermitian(self.UTx,self.SigmaTx)
        self.V.mult(self.SigmaTx,y)
        return y
    
    def apply_hermitian_transpose_mat(self, X, Y=None, intermediate_mats=None):
        if intermediate_mats == None:
            destroy = True
            UTX, STX = self.create_intermediate_matrices(X.getSizes())
        else:
            destroy = False
            UTX, STX = intermediate_mats
        self.U.hermitianTranspose()
        self.Sigma.hermitianTranspose()
        self.U.matMult(X, UTX)
        self.Sigma.matMult(UTX, STX)
        Y = self.V.matMult(STX, Y)
        self.U.hermitianTranspose()
        self.Sigma.hermitianTranspose()
        if destroy:
            UTX.destroy()
            STX.destroy()
        return Y
    
    def destroy_intermediate_vectors(self):
        self.VTx.destroy()
        self.Sigmax.destroy()
        self.SigmaTx.destroy()
        self.UTx.destroy()

    def destroy(self):
        self.destroy_intermediate_vectors()
        self.U.destroy()
        self.Sigma.destroy()
        self.V.destroy()