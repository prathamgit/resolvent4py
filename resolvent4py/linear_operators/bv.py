from .linear_operator import LinearOperator
from .. import MPI
from .. import PETSc
from .. import np
from ..miscellaneous import compute_local_size
from ..comms import distributed_to_sequential_vector
from ..comms import sequential_to_distributed_vector
from ..comms import distributed_to_sequential_matrix
from ..comms import sequential_to_distributed_matrix

class BvLinearOperator(LinearOperator):
    r"""
        Class for a linear operator of the form

        .. math::

            L = U \Sigma,

        where :math:`U` and :math:`\Sigma` are matrices of
        conformal sizes (and :math:`\Sigma` is not necessarily diagonal).

        :param comm: MPI communicator (one of :code:`MPI.COMM_WORLD` or
            :code:`MPI.COMM_SELF`)
        :param U: a SLEPc BV
        :type U: `BV`_
        :param Sigma: a SLEPc BV
        :type Sigma: `BV`_
        :param nblocks: [optional] number of blocks (if the linear operator \
            has block structure)
        :type nblocks: int
    """
    def __init__(self, comm, U, Sigma, nblocks=None):
        self.U = U
        self.Sigma = Sigma
        dims = (U.getSizes()[0], \
                (compute_local_size(Sigma.shape[-1]), Sigma.shape[-1]))
        super().__init__(comm, 'BvLinearOperator', dims, nblocks)

    def apply(self, x, y=None):
        y = self.create_left_vector() if y == None else y
        xseq = distributed_to_sequential_vector(self._comm, x)
        self.U.multVec(1.0, 0.0, y, self.Sigma@xseq.getArray())
        xseq.destroy()
        return y
    
    def apply_hermitian_transpose(self, x, y=None):
        y = self.create_right_vector() if y == None else y
        q = self.Sigma.conj().T@self.U.dotVec(x)
        qvec = PETSc.Vec().createWithArray(q, len(q), None, MPI.COMM_SELF)
        y = sequential_to_distributed_vector(qvec, y)
        return y
    
    def apply_mat(self, X, Y=None):
        M = distributed_to_sequential_matrix(self._comm, X)
        L = self.Sigma@M.getDenseArray()
        Lm = PETSc.Mat().createDense(L.shape, None, L, MPI.COMM_SELF)
        Y = self.create_left_bv(X.getSizes()[-1]) if Y == None else Y
        Y.mult(1.0, 0.0, self.U, Lm)
        Lm.destroy()
        M.destroy()
        return Y
    
    def apply_hermitian_transpose_mat(self, X, Y=None):
        M = X.dot(self.U)
        Ma = M.getDenseArray()
        sz = (self.Sigma[-1], self.X.getSizes()[-1])
        Y = np.zeros(sz, Ma.dtype) if Y == None else Y
        Y[:, :] = self.Sigma.conj().T@Ma
        M.destroy()
        return Y

    def destroy(self):
        self.U.destroy()
        del self.Sigma