import abc
import numpy as np
from petsc4py import PETSc
from ..error_handling_functions import raise_not_implemented_error
from ..petsc4py_helper_functions import generate_random_petsc_vector
from ..petsc4py_helper_functions import enforce_complex_conjugacy



class LinearOperator(metaclass=abc.ABCMeta):
    r"""
        Abstract base class for linear operators :math:`L`

        :param comm: MPI communicator (one of :code:`MPI.COMM_WORLD` or
            :code:`MPI.COMM_SELF`)
        :param name: name of the linear operator
        :type name: str
        :param dimensions: row and column sizes of the linear operator
        :type dimensions: `MatSizeSpec`_
        :param nblocks: number of blocks (if the linear operator has block \
            structure)
        :type nblocks: int
    """

    def __init__(self, comm, name, dimensions, nblocks):
        self._comm = comm
        self._name = name
        self._dimensions = dimensions
        self._nblocks = nblocks

    def get_comm(self):
        """The MPI communicator"""
        return self._comm
    
    def get_name(self):
        """The name of the linear operator"""
        return self._name
    
    def get_dimensions(self):
        """The dimensions of the linear operator"""
        return self._dimensions
    
    def get_nblocks(self):
        """The number of blocks of the linear operator"""
        return self._nblocks
    
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
    
    def check_if_real_valued(self):
        r"""
            :return: :code:`True` if the linear operator is real-valued, 
                :code:`False` otherwise
            :rtype: bool
        """
        sizes = self.get_dimensions()[-1]
        array = np.random.randn(sizes[0])
        x = PETSc.Vec().createWithArray(array, sizes, None, self.get_comm())
        Lx = self.apply(x)
        result = True if np.linalg.norm(Lx.getArray().imag) <= 1e-15 else False
        x.destroy()
        Lx.destroy()
        return result
    
    def check_if_complex_conjugate_structure(self):
        r"""
            Given a vector x

            .. math::
                x = \left(\ldots,x_{-1},x_{0},x_{1},\ldots\right)

            with vector-valued entries that satisfy \
            :math:`x_{-i} = \overline{x_i}`, check if the vector :math:`Lx` \
            also satisfies :math:`(Lx)_{-i}=\overline{(Lx)_{i}}`.

            :rtype: bool
        """
        x = generate_random_petsc_vector(self._comm, self._dimensions[-1])
        enforce_complex_conjugacy(self._comm, x, self._nblocks)
        Lx = self.apply(x)
        result = True if np.linalg.norm(Lx.getArray().imag) <= 1e-15 else False
        x.destroy()
        Lx.destroy()
        return result

    
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
