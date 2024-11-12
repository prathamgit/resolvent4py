from .linear_operator import LinearOperator
from .. import MPI
from .. import PETSc

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

    def apply(self, x, y=None):
        y = self.create_left_vector() if y == None else y
        q = self.Sigma@self.V.dotVec(x)
        self.U.multVec(1.0, 0.0, y, q)
        return y
    
    def apply_hermitian_transpose(self, x, y=None):
        y = self.create_right_vector() if y == None else y
        q = self.Sigma.conj().T@self.U.dotVec(x)
        self.V.multVec(1.0, 0.0, y, q)
        return y
    
    def apply_mat(self, X, Y=None):
        M = X.dot(self.V)
        L = self.Sigma@M.getDenseArray()
        Lm = PETSc.Mat().createDense(L.shape, None, L, MPI.COMM_SELF)
        Y = X.duplicate() if Y == None else Y
        Y.mult(1.0, 0.0, self.U, Lm)
        Lm.destroy()
        M.destroy()
        return Y
    
    def apply_hermitian_transpose_mat(self, X, Y=None):
        M = X.dot(self.U)
        L = self.Sigma.conj().T@M.getDenseArray()
        Lm = PETSc.Mat().createDense(L.shape, None, L, MPI.COMM_SELF)
        Y = X.duplicate() if Y == None else Y
        Y.mult(1.0, 0.0, self.V, Lm)
        Lm.destroy()
        M.destroy()
        return Y

    def destroy(self):
        self.U.destroy()
        self.V.destroy()
        del self.Sigma