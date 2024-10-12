from petsc4py import PETSc
from .linear_operator import LinearOperator
from ..petsc4py_helper_functions import compute_dense_inverse
from ..petsc4py_helper_functions import mat_solve_hermitian_transpose

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
    """

    def __init__(self, comm, A, B, K, C, woodbury_factors=None):

        super().__init__(comm, 'LowRankUpdatedLinearOperator', \
                         A.get_dimensions())
        self.A = A
        self.B = B
        self.K = K
        self.C = C

        self.X, self.D, self.Y = None, None, None
        if woodbury_factors == None:
            try:
                self.compute_woodbury_factors()
            except:
                self.X, self.D, self.Y = None, None, None
            

    def compute_woodbury_factors(self):
        r"""
            Compute Woodbury factors :code:`X`, :code:`D` and :code:`Y` 
            and set corresponding attributes.

            :rtype: None
        """

        self.X = self.B.duplicate()
        self.A.ksp.matSolve(self.B,self.X)
        self.Y = mat_solve_hermitian_transpose(self.A.ksp,self.C)
        self.C.hermitianTranspose()
        M = self.C.matMult(self.X).matMult(self.K)
        Id = PETSc.Mat().createConstantDiagonal(M.getSizes(),1.0,\
                                                comm=self.get_comm())
        M.axpy(1.0,Id,structure=PETSc.Mat.Structure.SAME)
        self.D = self.K.matMult(compute_dense_inverse(self.get_comm(),M))
        self.C.hermitianTranspose()

    def apply_low_rank_factors(self, F1, F2, F3, x):
        r"""
            :param F1: a dense PETSc matrix
            :type F1: PETSc.Mat.Type.DENSE
            :param F2: a dense PETSc matrix
            :type F2: PETSc.Mat.Type.DENSE
            :param F3: a dense PETSc matrix
            :type F3: PETSc.Mat.Type.DENSE
            :param x: a PETSc vector
            :type x: `Vec`_

            :return: :math:`F_1 F_2 F_3^* x`
            :rtype: `Vec`_
        """
        z = F3.createVecRight()
        q = F2.createVecLeft()
        y = F1.createVecLeft()
        F3.multHermitian(x,z)
        F2.mult(z,q)
        F1.mult(q,y)
        return y
    
    def apply_low_rank_factors_hermitian_transpose(self, F1, F2, F3, x):
        r"""
            :param F1: a dense PETSc matrix
            :type F1: PETSc.Mat.Type.DENSE
            :param F2: a dense PETSc matrix
            :type F2: PETSc.Mat.Type.DENSE
            :param F3: a dense PETSc matrix
            :type F3: PETSc.Mat.Type.DENSE
            :param x: a PETSc vector
            :type x: `Vec`_

            :return: :math:`F_3 F_2^* F_1^* x`
            :rtype: `Vec`_
        """
        z = F1.createVecRight()
        q = F2.createVecRight()
        y = F3.createVecLeft()
        F1.multHermitian(x,z)
        F2.multHermitian(z,q)
        F3.mult(q,y)
        return y

    def apply(self, x):
        y = self.A.apply(x)
        z = self.apply_low_rank_factors(self.B,self.K,self.C,x)
        y.axpy(1.0,z)
        return y
    
    def apply_hermitian_transpose(self, x):
        y = self.A.apply_hermitian_transpose(x)
        z = self.apply_low_rank_factors_hermitian_transpose(self.B,\
                                                            self.K,\
                                                            self.C,\
                                                            x)
        y.axpy(1.0,z)
        return y
    
    def solve(self, x):
        y = self.A.solve(x)
        z = self.apply_low_rank_factors(self.X,self.D,self.Y,x)
        y.axpy(-1.0,z)
        return y
        
    def solve_hermitian_transpose(self, x):
        y = self.A.solve_hermitian_transpose(x)
        z = self.apply_low_rank_factors_hermitian_transpose(self.X,\
                                                            self.D,
                                                            self.Y,\
                                                            x)
        y.axpy(-1.0,z)
        return y
    
    def destroy(self):
        r"""
            Destroys all attributes of the class, except for :code:`A`,
            which is itself a class with its own :code:`destroy()` method.
        """
        self.B.destroy()
        self.K.destroy()
        self.C.destroy()

        if self.X != None:
            self.X.destroy()
            self.D.destroy()
            self.Y.destroy()
