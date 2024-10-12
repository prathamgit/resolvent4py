from .linear_operator import LinearOperator

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
    """
    
    def __init__(self, comm, U, Sigma, V):
        
        dimensions = (U.getSizes()[0],V.getSizes()[0])
        super().__init__(comm, 'LowRankLinearOperator', dimensions)
        self.U = U
        self.Sigma = Sigma
        self.V = V

    def apply(self, x):
        z = self.V.createVecRight()
        q = self.Sigma.createVecLeft()
        y = self.U.createVecLeft()
        self.V.multHermitian(x,z)
        self.Sigma.mult(z,q)
        self.U.mult(q,y)
        return y
    
    def apply_low_rank_factors_hermitian_transpose(self, x):
        z = self.U.createVecRight()
        q = self.Sigma.createVecRight()
        y = self.V.createVecLeft()
        self.U.multHermitian(x,z)
        self.Sigma.multHermitian(z,q)
        self.V.mult(q,y)
        return y
    
    def destroy(self):
        self.U.destroy()
        self.Sigma.destroy()
        self.V.destroy()