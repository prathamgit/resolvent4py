import abc
from petsc4py import PETSc
from ..error_handling_functions import raise_not_implemented_error

class LinearOperator(metaclass=abc.ABCMeta):
    r"""
        Abstract base class for linear operators :math:`L`

        :param comm: MPI communicator (one of :code:`MPI.COMM_WORLD` or
            :code:`MPI.COMM_SELF`)
        :param name: name of the linear operator
        :type name: str
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
