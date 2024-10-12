import abc

from petsc4py import PETSc
from mpi4py import MPI
import scipy as sp
import numpy as np

from .error_handling_functions import raise_not_implemented_error
from .petsc4py_helper_functions import compute_dense_inverse
from .petsc4py_helper_functions import mat_solve_hermitian_transpose

class LinearOperator(metaclass=abc.ABCMeta):
    r"""
        Abstract base class for linear operators :math:`L`

        :param comm: MPI communicator (one of :code:`MPI.COMM_WORLD` or
            :code:`MPI.COMM_SELF`)
        :param dimensions: row and column sizes of the linear operator
        :type dimensions: `MatSizeSpec`_
    """

    def __init__(self, comm, name, dimensions):
        self._comm = comm
        self._name = name
        self._dimensions = dimensions

    def get_comm(self):
        """The MPI communicator"""
        return self._comm
    
    def get_name(self):
        """The name of the linear operator"""
        return self._name
    
    def get_dimensions(self):
        """The dimensions of the linear operator"""
        return self._dimensions
    
    def create_right_vector(self):
        r"""
            :return: a PETSc vector that :math:`L` can be multiplied against
            :rtype: `Vec`_
        """
        vec = PETSc.Vec().create(comm=self._comm)
        vec.setSizes(self._dimensions[-1])
        vec.setUp()
        return vec

    def create_left_vector(self):
        r"""
            :return: a PETSc vector where :math:`Lx` can be stored into
            :rtype: `Vec`_
        """
        vec = PETSc.Vec().create(comm=self._comm)
        vec.setSizes(self._dimensions[0])
        vec.setUp()
        return vec

    # Methods that must be implemented by subclasses
    @abc.abstractmethod
    def apply(self,x):
        r"""
            :param x: a PETSc vector
            :type x: `Vec`_

            :return: :math:`Lx`
            :rtype: `Vec`_
        """

    @abc.abstractmethod
    def destroy(self):
        r"""
            Destroy the PETSc objects associated with :math:`L`
        """
    
    # Methods that don't necessarily need to be implemented by subclasses
    @abc.abstractmethod
    def apply_hermitian_transpose(self,x):
        r"""
            :param x: a PETSc vector
            :type x: `Vec`_

            :return: :math:`L^* x`
            :rtype: `Vec`_
        """

    @raise_not_implemented_error
    def solve(self, x):
        r"""
            :param x: a PETSc vector
            :type x: `Vec`_

            :return: :math:`L^{-1}x`
            :rtype: `Vec`_
        """
    
    @raise_not_implemented_error
    def solve_hermitian_transpose(self, x):
        r"""
            :param x: a PETSc vector
            :type x: `Vec`_

            :return: :math:`L^{-*}x`
            :rtype: `Vec`_
        """

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class MatrixLinearOperator(LinearOperator):
    r"""
        Class for a linear operator :math:`L = A`, where :math:`A` is a matrix.
        In general, :code:`A` can be any matrix (rectangular or square, 
        invertible or non-invertible). If a :code:`ksp` is provided, however, 
        :code:`A` must be invertible.

        :param comm: MPI communicator (one of :code:`MPI.COMM_WORLD` or
            :code:`MPI.COMM_SELF`)
        :param A: a PETSc matrix
        :type A: `MatType`_
        :param ksp: [optional] a PETSc KSP object to enable the :code:`solve()`
            and :code:solve_hermitian_transpose()` methods. If a KSP is 
            provided, its type and the type of A must be compatible.
        :type ksp: `KSP`_
    """

    def __init__(self, comm, A, ksp=None):

        super().__init__(comm, 'MatrixLinearOperator', A.getSizes())
        self.A = A
        self.ksp = ksp
        
    def apply(self, x):
        y = self.create_left_vector()
        self.A.mult(x,y)
        return y
    
    def apply_hermitian_transpose(self, x):
        y = self.create_right_vector()
        self.A.multHermitian(x,y)
        return y
    
    def solve(self, x):
        if self.ksp != None:
            y = self.create_left_vector()
            self.ksp.solve(x,y)
            return y
        else:
            raise Exception(
                f"Error from {self.get_name()}.solve(): "
                f"Please provide a PETSc KSP object at initialization to use "
                f"the solve() method."
            )
        
    def solve_hermitian_transpose(self, x):
        if self.ksp != None:
            y = self.create_right_vector()
            x.conjugate()
            self.ksp.solveTranspose(x,y)
            x.conjugate()
            y.conjugate()
            return y
        else:
            raise Exception(
                f"Error from {self.get_name()}.solve_hermitian_transpose(). "
                f"Please provide a PETSc KSP object at initialization to use "
                f"the solve_hermitian_transpose() method."
            )
        
    def destroy(self):
        self.A.destroy()
        self.ksp.destroy() if self.ksp != None else None

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class SumOfLowRankOperatorsLinearOperator(LinearOperator):
    r"""
        Class for a linear operator of the form

        .. math::

            L = A + U \Sigma V^*,

        where :math:`U`, :math:`\Sigma` and :math:`V` are matrices of
        conformal sizes (and :math:`\Sigma` is not necessarily diagonal) and 
        :math:`A` is an instance of the class
        :class:`resolvent4py.linear_operators.LowRankLinearOperator`

        :param comm: MPI communicator (one of :code:`MPI.COMM_WORLD` or
            :code:`MPI.COMM_SELF`)
        :param A: instance of the \
            :class:`resolvent4py.linear_operators.LowRankLinearOperator` class
        :param U: dense PETSc matrix
        :type U: PETSc.Mat.Type.DENSE
        :param Sigma: dense PETSc matrix
        :type Sigma: PETSc.Mat.Type.DENSE
        :param V: dense PETSc matrix
        :type V: PETSc.Mat.Type.DENSE
    """

    def __init__(self, comm, A, U, Sigma, V):
        
        super().__init__(comm, 'SumOfLowRankOperatorsLinearOperator', \
                         self.A.get_dimensions())
        self.A = A
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
        y.axpy(1.0,self.A.apply(x))
        return y
    
    def apply_low_rank_factors_hermitian_transpose(self, x):
        z = self.U.createVecRight()
        q = self.Sigma.createVecRight()
        y = self.V.createVecLeft()
        self.U.multHermitian(x,z)
        self.Sigma.multHermitian(z,q)
        self.V.mult(q,y)
        y.axpy(1.0,self.A.apply_hermitian_transpose(x))
        return y
    
    def destroy(self):
        r"""
            Destroys all attributes of the class, except for :code:`A`,
            which is itself a class with its own :code:`destroy()` method.
        """
        self.U.destroy()
        self.Sigma.destroy()
        self.V.destroy()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class LowRankUpdatedLinearOperator(LinearOperator):
    r"""
        Class for a linear operator of the form

        .. math::

            L = A + B K C^*
        
        where :math:`A` is an instance of any subclass of the 
        :class:`resolvent4py.linear_operators.LinearOperator` class,
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
            :class:`resolvent4py.linear_operators.LinearOperator` class
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

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class ProjectedLinearOperator(LinearOperator):
    r"""
        Class for a linear operator of the form :math:`L = PAP`,
        where :math:`A` is an instance of any subclass of the 
        :class:`resolvent4py.linear_operators.LinearOperator` class,
        and :math:`P` is a projection operator of the form

        .. math::

            P = \begin{cases} 
            \Phi\left(\Psi^*\Phi\right)^{-1}\Psi^* & \text{if }\ 
                \text{complement} = \text{False} \\
            I - \Phi\left(\Psi^*\Phi\right)^{-1}\Psi^*  & \text{if }\ 
                \text{complement} = \text{True}
            \end{cases}

        If the :code:`A.solve()` method is enabled, then the :code:`solve()` 
        method for this class returns

        .. math::

            y = P \circ f \circ Px,
        
        where :math:`f(z)` denotes the output of :code:`A.solve(z)`.


        :param comm: MPI communicator (one of :code:`MPI.COMM_WORLD` or
            :code:`MPI.COMM_SELF`)
        :param A: instance of a subclass of the
            :class:`resolvent4py.linear_operators.LinearOperator` class
        :param Phi: dense PETSc matrix
        :type Phi: PETSc.Mat.Type.DENSE
        :param Psi: dense PETSc matrix
        :type Psi: PETSc.Mat.Type.DENSE
        :param complement: specifies which projection operator
        :type complement: Bool
    """
    
    def __init__(self, comm, A, Phi, Psi, complement=False):

        super().__init__(comm, 'ProjectedLinearOperator', A.get_dimensions())
        self.A = A
        self.Phi = Phi
        self.Psi = Psi
        self.complement = complement
        self.enforce_biorthogonality()

    def enforce_biorthogonality(self):
        r"""
            Enforce biorthogonality
            :math:`\Phi\leftarrow \Phi\left(\Psi^*\Phi\right)^{-1}`.

            :rtype: None
        """
        self.Psi.conjugate()
        self.Phi = self.Phi.matMult(compute_dense_inverse(\
            self.get_comm(),self.Psi.transposeMatMult(self.Phi)))
        self.Psi.conjugate()
    
    def apply_projection(self, x):
        r"""            
            :param x: a PETSc vector
            :type x: PETSc.Vec

            :return: :math:`P x`
            :rtype: PETSc.Vec
        """
        z, y = self.Phi.createVecs()
        if self.complement:
            self.Psi.multHermitian(x,z)
            self.Phi.mult(z,y)
            y.axpy(-1.0,x)
            y.scale(-1.0)
        else:
            self.Psi.multHermitian(x,z)
            self.Phi.mult(z,y)
        return y
    
    def apply_projection_hermitian_transpose(self, x):
        r"""            
            :param x: a PETSc vector
            :type x: PETSc.Vec

            :return: :math:`P^* x`
            :rtype: PETSc.Vec
        """
        z, y = self.Phi.createVecs()
        if self.complement:
            self.Phi.multHermitian(x,z)
            self.Psi.mult(z,y)
            y.axpy(-1.0,x)
            y.scale(-1.0)
        else:
            self.Phi.multHermitian(x,z)
            self.Psi.mult(z,y)
        return y
    
    def apply(self, x):
        return self.apply_projection(self.A.apply(self.apply_projection(x)))

    def apply_hermitian_transpose(self, x):
        return self.apply_projection_hermitian_transpose(\
                self.A.apply_hermitian_transpose(\
                self.apply_projection_hermitian_transpose(x)))
    
    def solve(self, x):
        return self.apply_projection(self.A.solve(self.apply_projection(x)))
    
    def solve_hermitian_transpose(self, x):
        return self.apply_projection_hermitian_transpose(\
                self.A.solve_hermitian_transpose(\
                self.apply_projection_hermitian_transpose(x)))

    def destroy(self):
        r"""
            Destroys all attributes of the class, except for :code:`A`,
            which is itself a class with its own :code:`destroy()` method.
        """
        self.Phi.destroy()
        self.Psi.destroy()